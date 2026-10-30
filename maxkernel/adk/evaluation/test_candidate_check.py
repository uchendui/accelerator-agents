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
])
def test_a_candidate_that_reaches_into_the_harness_is_flagged(source):
  assert harness_tampering(source) is not None


def test_an_ordinary_pallas_kernel_passes():
  source = (ADK / "evaluation/jaxbench_adapted_dataset/level2/8p_GEMM/reference.py").read_text(encoding="utf-8")
  assert harness_tampering(source) is None
  assert harness_tampering("def computation(x: int) -> int:\n    return x * 2\n") is None


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
