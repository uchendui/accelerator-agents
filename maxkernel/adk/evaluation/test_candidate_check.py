from pathlib import Path

import pytest

from evaluation.candidate_check import harness_tampering

ADK = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("source", [
  'import sys\nsys.modules["base_kernel"].HARNESS_TIMER = "wallclock"\n',
  "import time\ntime.perf_counter = lambda: 0.0\n",
  "import jax\njax.profiler.trace = None\n",
  "import time\nsetattr(time, 'perf_counter', lambda: 0.0)\n",
  "import importlib\nimportlib.import_module('base_kernel').HARNESS_TIMER = 'wallclock'\n",
  "from auto_agent.tools import test_harness\n",
  "import evaluation.harness_code\n",
  'vars(__import__("sys").modules["__main__"])["HARNESS_" + "TIMER"] = "wallclock"\n',
  '__import__("jax").profiler.trace = None\n',
  "import importlib\n",
  'globals()["HARNESS_" + "TIMER"] = "wallclock"\n',
  'import jax\ngetattr(jax, "prof" + "iler").trace = None\n',
  "import sys\nframe = sys._getframe(1)\n",
  "import jax\njax.__dict__['profiler'] = None\n",
])
def test_a_candidate_that_reaches_into_the_harness_is_flagged(source):
  assert harness_tampering(source) is not None


@pytest.mark.parametrize("source", [
  (ADK / "evaluation/jaxbench_adapted_dataset/level2/8p_GEMM/reference.py").read_text(encoding="utf-8"),
  "def computation(x: int) -> int:\n    return x * 2\n",
  # A Pallas kernel writes its output and accumulator refs, and may read a constant-named attribute.
  "import jax\n\ndef kernel(x_ref, o_ref, acc_ref):\n    acc_ref[...] += x_ref[...]\n    o_ref[...] = acc_ref[...]\n"
  "    cfg = {}\n    cfg['block'] = getattr(jax, 'numpy')\n",
  # The two lines that flagged 32 of 37 real kernels: the usual main block, and a write through a ref method.
  'def computation(x):\n    return x\n\n\nif __name__ == "__main__":\n    print(computation(1))\n',
  "def kernel(x_ref, out_ref):\n    out_ref.reshape(1, 8)[...] = x_ref[...]\n",
])
def test_an_ordinary_pallas_kernel_passes(source):
  assert harness_tampering(source) is None


def test_the_evaluator_fails_a_tampering_candidate_before_the_harness_runs(tmp_path, monkeypatch):
  from types import SimpleNamespace

  from evaluation import jax_kernel_evaluator

  optimized = tmp_path / "optimized.py"
  optimized.write_text('import sys\nsys.modules["reference"].HARNESS_TIMER = "wallclock"\n', encoding="utf-8")
  monkeypatch.setattr(jax_kernel_evaluator, "load_kernel_task_from_yaml", lambda path: SimpleNamespace(task_id="t"))
  monkeypatch.setattr(jax_kernel_evaluator.tempfile, "mkdtemp", lambda **kwargs: pytest.fail("the harness ran"))

  result = jax_kernel_evaluator.JAXKernelEvaluator(local=True).evaluate(
    str(tmp_path / "reference.py"), str(optimized), str(tmp_path / "task.yaml"))

  assert result.task_id == "t"
  assert result.error_trace == [f"harness tampering: {jax_kernel_evaluator.harness_tampering(optimized.read_text())}"]
  assert result.compiled_successfully == [] and result.speedup == []
