import string

HARNESS_TEMPLATE = string.Template("""
import time
import json
import jax
import jax.numpy as jnp
import importlib
import importlib.util
import os
import sys
import traceback


def load_module_from_path(module_name, file_path):
  spec = importlib.util.spec_from_file_location(module_name, file_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Could not load {module_name} from {file_path}")
  module = importlib.util.module_from_spec(spec)
  # Register before exec_module: code that inspects sys.modules[cls.__module__]
  # (e.g. dataclasses resolving string annotations) fails otherwise.
  sys.modules[module_name] = module
  try:
    spec.loader.exec_module(module)
  except Exception:
    sys.modules.pop(module_name, None)
    raise
  return module


def normalize_tolerance(value):
  if isinstance(value, list):
    return [float(item) for item in value]
  return float(value)


def validate_output_tolerances(output_count, atol, rtol):
  # A list holds one tolerance per output leaf in tree_leaves order.
  for name, tolerance in (("atol", atol), ("rtol", rtol)):
    if isinstance(tolerance, list) and len(tolerance) != output_count:
      raise ValueError(f"{name} list length ({len(tolerance)}) does not match output count ({output_count})")


def outputs_match(expected, actual, atol, rtol):
  if len(expected) != len(actual):
    return False
  return all(
      b.shape == o.shape and bool(jnp.allclose(
          b, o,
          atol=atol[i] if isinstance(atol, list) else atol,
          rtol=rtol[i] if isinstance(rtol, list) else rtol))
      for i, (b, o) in enumerate(zip(expected, actual)))


def benchmark(func, args, static_argnums, timer, num_iters=50, num_warmups=5):
  # The search harness's timer (auto_agent/tools/test_harness.py benchmark), so zero-shot and
  # search speedups are measured the same way: 5 warmups, then the device time of each of 50 runs
  # of the jitted program read from a jax.profiler trace, median. A trace without exactly one
  # device event per run raises. timer is the reference's HARNESS_TIMER: "wallclock" times each
  # run with perf_counter instead (kda's many-small-op reference defeats the profiler).
  import glob
  import shutil
  import statistics
  import tempfile

  if timer not in ("device", "wallclock"):
    raise ValueError("reference.HARNESS_TIMER must be 'device' or 'wallclock', got " + repr(timer))

  dynamic_args = tuple(arg for i, arg in enumerate(args) if i not in static_argnums)

  def benchmark_func(*f_args):
    all_args = list(args)
    dyn_idx = 0
    for i in range(len(args)):
      if i not in static_argnums:
        all_args[i] = f_args[dyn_idx]
        dyn_idx += 1
    return func(*all_args)

  compiled_func = jax.jit(benchmark_func, static_argnums=static_argnums).lower(*args).compile()
  for _ in range(num_warmups):
    jax.block_until_ready(compiled_func(*dynamic_args))

  if timer == "wallclock":
    times = []
    for _ in range(num_iters):
      start = time.perf_counter()
      jax.block_until_ready(compiled_func(*dynamic_args))
      times.append(time.perf_counter() - start)
    return statistics.median(times)

  trace_dir = tempfile.mkdtemp(prefix="harness_trace_")
  try:
    with jax.profiler.trace(trace_dir):
      for _ in range(num_iters):
        jax.block_until_ready(compiled_func(*dynamic_args))
    xplanes = glob.glob(trace_dir + "/**/*.xplane.pb", recursive=True)
    if len(xplanes) != 1:
      raise RuntimeError("expected one xplane.pb under " + trace_dir + ", found " + str(len(xplanes)))
    profile = jax.profiler.ProfileData.from_file(xplanes[0])
    times = [event.duration_ns / 1e9
             for plane in profile.planes if plane.name.startswith("/device:")
             for line in plane.lines if line.name == "XLA Modules"
             for event in line.events
             if event.duration_ns > 0 and event.name.startswith("jit_benchmark_func(")]
  finally:
    shutil.rmtree(trace_dir, ignore_errors=True)

  if len(times) != num_iters:
    raise RuntimeError("expected " + str(num_iters) + " jit_benchmark_func device events in the "
                       "profiler trace, found " + str(len(times)))
  return statistics.median(times)


def main():
  try:
    # Load task configuration from task.json
    if not os.path.exists("task.json"):
      raise FileNotFoundError(
          "task.json not found. It should contain input_gen_code.")

    with open("task.json", "r") as f:
      task_data = json.load(f)

    input_gen_code = task_data.get("input_gen_code")
    task_atol = normalize_tolerance(task_data.get("atol", 1e-3))
    task_rtol = normalize_tolerance(task_data.get("rtol", 1e-3))
    sort_outputs = task_data.get("sort_outputs", False)
    if not isinstance(sort_outputs, bool):
      raise TypeError("sort_outputs must be a boolean")

    if input_gen_code:
      ldict = {}
      try:
        exec(input_gen_code, globals(), ldict)
      except Exception as e:
        raise RuntimeError(f"Failed to execute input_gen_code: {e}")

      if "get_inputs" not in ldict:
        raise RuntimeError("input_gen_code must define get_inputs()")

      try:
        raw_inputs = ldict["get_inputs"]()
      except Exception as e:
        raise RuntimeError(f"Error while running get_inputs(): {e}")

      # Check if the input is a list of tuples or a single tuple
      multiple_input_configs = isinstance(raw_inputs, list)
      if multiple_input_configs:
        inputs_list = raw_inputs
      elif isinstance(raw_inputs, tuple) and len(raw_inputs) == 2:
        inputs_list = [raw_inputs]
      else:
        raise ValueError(
            f"get_inputs() must return a list of tuples or a single (dynamic_args, static_args) tuple. Got: {type(raw_inputs)}"
        )
    else:
      raise ValueError("input_gen_code must be provided.")

    # Import the uploaded scripts as modules
    base_mod = load_module_from_path("reference", "reference.py")
    optimized_mod = load_module_from_path("optimized", "optimized.py")
    timer = getattr(base_mod, "HARNESS_TIMER", "device")

    harness_logs = []

    # reference_time_ms / optimized_time_ms hold the search harness timer's median (see benchmark).
    result = {
        "compiled_successfully": [],
        "numerically_correct": [],
        "max_abs_diff": [],
        "max_rel_diff": [],
        "reference_time_ms": [],
        "optimized_time_ms": [],
        "error_trace": [],
    }

    # Iterate over all input configurations
    for idx, inputs in enumerate(inputs_list):
      if not isinstance(inputs, tuple) or len(inputs) != 2:
        raise ValueError(
            f"Each input config must return exactly 2 elements: (dynamic_args, static_args). Got: {type(inputs)}"
        )

      dynamic_args, static_args = inputs
      if not isinstance(dynamic_args, (list, tuple)) or not isinstance(static_args, (list, tuple)):
        raise TypeError("Both dynamic_args and static_args must be lists or tuples.")

      args = tuple(dynamic_args) + tuple(static_args)
      static_argnums = tuple(range(len(dynamic_args), len(args)))

      # A list indexes inputs for several configs, and outputs for one config.
      current_tolerances = {}
      for name, tolerance in (("atol", task_atol), ("rtol", task_rtol)):
        if multiple_input_configs and isinstance(tolerance, list):
          if len(tolerance) != len(inputs_list):
            raise ValueError(f"{name} list length ({len(tolerance)}) does not match input count ({len(inputs_list)})")
          current_tolerances[name] = tolerance[idx]
        else:
          current_tolerances[name] = tolerance
      curr_atol = current_tolerances["atol"]
      curr_rtol = current_tolerances["rtol"]

      # 1. Correctness Check
      try:
        jit_base = jax.jit(base_mod.computation, static_argnums=static_argnums)
        out_base = jax.block_until_ready(jit_base(*args))
        out_base_cpu = jax.device_get(out_base)
        del out_base
      except Exception as e:
        result["compiled_successfully"].append(False)
        result["error_trace"].append(f"Reference model failed: {traceback.format_exc()}")
        result["numerically_correct"].append(False)
        result["max_abs_diff"].append(None)
        result["max_rel_diff"].append(None)
        result["reference_time_ms"].append(0.0)
        result["optimized_time_ms"].append(0.0)
        continue

      # Dirty all HBM memory leaves with NaN / Sentinel values to prevent cache reuse
      try:
        for leaf in jax.tree_util.tree_leaves(out_base_cpu):
          if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            if jnp.issubdtype(leaf.dtype, jnp.floating) or jnp.issubdtype(leaf.dtype, jnp.complexfloating):
              val = jnp.nan
            elif jnp.issubdtype(leaf.dtype, jnp.bool_):
              val = True
            else:
              val = 123  # Fits within int8/uint8 and all larger integer dtypes

            dummy = jnp.full(leaf.shape, val, dtype=leaf.dtype)
            dummy.block_until_ready()
            del dummy
      except Exception as e:
        harness_logs.append(f"Failed to dirty HBM memory: {e}")

      try:
        jit_optimized = jax.jit(optimized_mod.computation, static_argnums=static_argnums)
        out_optimized = jax.block_until_ready(jit_optimized(*args))
        out_optimized_cpu = jax.device_get(out_optimized)
        del out_optimized
      except Exception as e:
        result["compiled_successfully"].append(False)
        result["error_trace"].append(traceback.format_exc())
        result["numerically_correct"].append(False)
        result["max_abs_diff"].append(None)
        result["max_rel_diff"].append(None)
        result["reference_time_ms"].append(0.0)
        result["optimized_time_ms"].append(0.0)
        continue

      result["compiled_successfully"].append(True)
      result["error_trace"].append(None)

      out_base_flat = jax.tree_util.tree_leaves(out_base_cpu)
      out_optimized_flat = jax.tree_util.tree_leaves(out_optimized_cpu)
      if sort_outputs:
        out_base_flat = [jnp.sort(output, axis=-1) for output in out_base_flat]
        out_optimized_flat = [jnp.sort(output, axis=-1) for output in out_optimized_flat]

      validate_output_tolerances(len(out_base_flat), curr_atol, curr_rtol)
      is_correct = True
      max_abs_diff = 0.0
      max_rel_diff = 0.0

      if len(out_base_flat) != len(out_optimized_flat):
        harness_logs.append(
            f"Output count mismatch for input {idx}: "
            f"expected {len(out_base_flat)}, got {len(out_optimized_flat)}")
      else:
        for output_idx, (b, o) in enumerate(zip(out_base_flat, out_optimized_flat)):
          if b.shape != o.shape:
            harness_logs.append(
                f"Shape mismatch for output {output_idx} of input {idx}: "
                f"expected {b.shape}, got {o.shape}")

      try:
        is_correct = outputs_match(
            out_base_flat, out_optimized_flat, curr_atol, curr_rtol)
        for b, o in zip(out_base_flat, out_optimized_flat):
          max_abs_diff = max(max_abs_diff, float(jnp.max(jnp.abs(b - o))))
          max_rel_diff = max(max_rel_diff, float(jnp.max(jnp.abs((b - o) / b))))
      except Exception as e:
        harness_logs.append(f"Correctness check failed for input {idx}: {e}")
        is_correct = False

      result["numerically_correct"].append(bool(is_correct))
      result["max_abs_diff"].append(max_abs_diff)
      result["max_rel_diff"].append(max_rel_diff)

      if not is_correct:
        result["reference_time_ms"].append(0.0)
        result["optimized_time_ms"].append(0.0)
        continue

      # Benchmark and collect timing results
      try:
        time_base = benchmark(base_mod.computation, args, static_argnums, timer)
        time_optimized = benchmark(optimized_mod.computation, args, static_argnums, timer)
        result["reference_time_ms"].append(time_base * 1000)
        result["optimized_time_ms"].append(time_optimized * 1000)
      except Exception as e:
        harness_logs.append(f"Benchmarking failed for input {idx}: {e}")
        result["reference_time_ms"].append(0.0)
        result["optimized_time_ms"].append(0.0)

    if harness_logs:
      result["logs"] = harness_logs
    with open("result.json", "w", encoding="utf-8") as f:
      json.dump(result, f)
  except Exception as e:
    with open("result.json", "w", encoding="utf-8") as f:
      json.dump({
          "error_trace": [traceback.format_exc()]
      }, f)


if __name__ == "__main__":
  main()
""")
