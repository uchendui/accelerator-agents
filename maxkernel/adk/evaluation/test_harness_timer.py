"""Both harnesses read the reference's timer before the candidate module runs, so a candidate that sets
HARNESS_TIMER on the reference at import is still timed on the device."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from evaluation.harness_code import HARNESS_TEMPLATE

ADK = Path(__file__).resolve().parents[1]
INPUTS = "import jax.numpy as jnp\n\ndef get_inputs():\n    return ([jnp.ones((8,))], [])\n"
TAMPERING = 'import sys\nsys.modules["{module}"].HARNESS_TIMER = "wallclock"\n\n\ndef computation(x):\n    return x * 2\n'
# A candidate that rebinds the test file's own global, which runs as __main__ in production.
REBINDING = 'import sys\nvars(sys.modules["harness_test"])["HARNESS_TIMER"] = "wallclock"\n\n\ndef computation(x):\n    return x * 2\n'


def load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def recording_benchmark(timers):
  def benchmark(func, args, static_argnums, timer, num_iters=50, num_warmups=5):
    timers.append(timer)
    return 1.0
  return benchmark


@pytest.mark.parametrize("candidate", [TAMPERING.format(module="base_kernel"), REBINDING], ids=["reference", "main"])
def test_the_search_harness_times_with_the_timer_read_before_the_candidate_imports(tmp_path, monkeypatch, candidate):
  template = load(ADK / "auto_agent/tools/test_harness.py", "search_harness_template").TEST_TEMPLATE
  (tmp_path / "base_kernel.py").write_text("def kernel(x):\n    return x * 2\n", encoding="utf-8")
  (tmp_path / "optimized_kernel.py").write_text(candidate, encoding="utf-8")
  (tmp_path / "harness_test.py").write_text(template.format(
    input_gen_code=INPUTS, kernel_name="kernel", atol=1e-3, rtol=1e-3, sort_before_compare=False), encoding="utf-8")
  for name in ("base_kernel", "optimized_kernel"):
    monkeypatch.delitem(sys.modules, name, raising=False)
  monkeypatch.syspath_prepend(str(tmp_path))

  harness = load(tmp_path / "harness_test.py", "harness_test")
  monkeypatch.setitem(sys.modules, "harness_test", harness)
  timers = []
  harness.benchmark = recording_benchmark(timers)
  harness.main()
  assert "wallclock" in (getattr(sys.modules["base_kernel"], "HARNESS_TIMER", ""), getattr(harness, "HARNESS_TIMER", ""))
  assert timers == ["device", "device"]


def test_the_eval_harness_times_with_the_timer_read_before_the_candidate_loads(tmp_path, monkeypatch):
  (tmp_path / "reference.py").write_text("def computation(x):\n    return x * 2\n", encoding="utf-8")
  (tmp_path / "optimized.py").write_text(TAMPERING.format(module="reference"), encoding="utf-8")
  (tmp_path / "task.json").write_text(json.dumps({"input_gen_code": INPUTS}), encoding="utf-8")
  (tmp_path / "harness.py").write_text(HARNESS_TEMPLATE.template, encoding="utf-8")
  for name in ("reference", "optimized"):
    monkeypatch.delitem(sys.modules, name, raising=False)
  monkeypatch.chdir(tmp_path)

  harness = load(tmp_path / "harness.py", "zeroshot_harness")
  timers = []
  harness.benchmark = recording_benchmark(timers)
  harness.main()
  assert sys.modules["reference"].HARNESS_TIMER == "wallclock"
  assert timers == ["device", "device"]
