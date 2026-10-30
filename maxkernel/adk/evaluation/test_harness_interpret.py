"""Both harnesses fail a candidate whose pallas_call runs with interpret set at run time, and leave the
reference's own interpret calls alone."""

import importlib.util
import json
import sys
from pathlib import Path

import jax.experimental.pallas
import pytest

from evaluation.harness_code import HARNESS_TEMPLATE

ADK = Path(__file__).resolve().parents[1]
# The candidate turns interpret mode on for a small input only, so the static check cannot see it.
CANDIDATE = (
  "import jax.experimental.pallas as pl\n\n\ndef kernel(x_ref, o_ref):\n    o_ref[...] = x_ref[...] * 2\n\n\n"
  "def computation(x):\n    return pl.pallas_call(kernel, out_shape=x, interpret=x.shape[0] < 128)(x)\n"
)
# The reference uses interpret mode outright, as a CPU task's reference may.
REFERENCE = (
  "import jax.experimental.pallas as pl\n\n\ndef kernel(x_ref, o_ref):\n    o_ref[...] = x_ref[...] * 2\n\n\n"
  "def {name}(x):\n    return pl.pallas_call(kernel, out_shape=x, interpret=True)(x)\n"
)
INPUTS = "import jax.numpy as jnp\n\ndef get_inputs():\n    return ([jnp.ones(({size},))], [])\n"


def load(path, name):
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def fake_pallas_call(kernel, **kwargs):
  """Stands in for the real pallas_call, which has no CPU lowering here: the kernels above double their input."""
  return lambda x: x * 2


def recording_benchmark(timers):
  def benchmark(func, args, static_argnums, timer, num_iters=50, num_warmups=5):
    timers.append(timer)
    return 1.0
  return benchmark


@pytest.mark.parametrize("size, interpreted", [(8, True), (128, False)])
def test_the_search_harness_fails_a_candidate_that_runs_pallas_in_interpret_mode(tmp_path, monkeypatch, capsys, size, interpreted):
  monkeypatch.setattr(jax.experimental.pallas, "pallas_call", fake_pallas_call)
  template = load(ADK / "auto_agent/tools/test_harness.py", "search_harness_template").TEST_TEMPLATE
  (tmp_path / "base_kernel.py").write_text(REFERENCE.format(name="kernel"), encoding="utf-8")
  (tmp_path / "optimized_kernel.py").write_text(CANDIDATE, encoding="utf-8")
  (tmp_path / "harness_test.py").write_text(template.format(
    input_gen_code=INPUTS.format(size=size), kernel_name="kernel", atol=1e-3, rtol=1e-3, sort_before_compare=False),
    encoding="utf-8")
  for name in ("base_kernel", "optimized_kernel"):
    monkeypatch.delitem(sys.modules, name, raising=False)
  monkeypatch.syspath_prepend(str(tmp_path))

  harness = load(tmp_path / "harness_test.py", "harness_test")
  timers = []
  harness.benchmark = recording_benchmark(timers)
  if interpreted:
    with pytest.raises(SystemExit):
      harness.main()
    assert "interpret mode: pallas_call ran with interpret=True" in capsys.readouterr().out
    assert timers == []
  else:
    harness.main()
    assert "CORRECTNESS: True" in capsys.readouterr().out
    assert timers == ["device", "device"]


@pytest.mark.parametrize("size, interpreted", [(8, True), (128, False)])
def test_the_eval_harness_fails_a_candidate_that_runs_pallas_in_interpret_mode(tmp_path, monkeypatch, size, interpreted):
  monkeypatch.setattr(jax.experimental.pallas, "pallas_call", fake_pallas_call)
  (tmp_path / "reference.py").write_text(REFERENCE.format(name="computation"), encoding="utf-8")
  (tmp_path / "optimized.py").write_text(CANDIDATE, encoding="utf-8")
  (tmp_path / "task.json").write_text(json.dumps({"input_gen_code": INPUTS.format(size=size)}), encoding="utf-8")
  (tmp_path / "harness.py").write_text(HARNESS_TEMPLATE.template, encoding="utf-8")
  for name in ("reference", "optimized"):
    monkeypatch.delitem(sys.modules, name, raising=False)
  monkeypatch.chdir(tmp_path)

  harness = load(tmp_path / "harness.py", "zeroshot_harness")
  timers = []
  harness.benchmark = recording_benchmark(timers)
  harness.main()
  result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
  if interpreted:
    assert result["compiled_successfully"] == [False]
    assert "interpret mode: pallas_call ran with interpret=True" in result["error_trace"][0]
    assert timers == []
  else:
    assert result["compiled_successfully"] == [True] and result["numerically_correct"] == [True]
    assert timers == ["device", "device"]
