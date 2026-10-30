import asyncio

import pytest

from auto_agent.client_utils import eval_client


def run_until_available(monkeypatch, outcomes):
  """Runs call_eval_server_until_available over a scripted eval server: each outcome is a result dict or
  an exception to raise, in call order. Returns the result and the waits slept between the calls."""
  calls = iter(outcomes)
  sleeps = []

  async def fake_call(session, url, payload, **kwargs):
    outcome = next(calls)
    if isinstance(outcome, BaseException):
      raise outcome
    return outcome

  async def fake_sleep(seconds):
    sleeps.append(seconds)

  monkeypatch.setattr(eval_client, "call_eval_server_async", fake_call)
  monkeypatch.setattr(eval_client.asyncio, "sleep", fake_sleep)
  result = asyncio.run(
    eval_client.call_eval_server_until_available(None, "http://localhost:1", {"timeout": 1}, "node_001")
  )
  return result, sleeps


SERVER_DOWN = "Failed to submit task. Status: 503, Error: Every TPU server of type 'tpu' is down"
TASK_FAILED_DOWN = "Task abc ended with status failed: Every TPU server of type 'tpu' is down"


def test_an_unavailable_eval_server_is_waited_out_with_a_doubling_wait(monkeypatch):
  result, sleeps = run_until_available(
    monkeypatch, [Exception(SERVER_DOWN), Exception(TASK_FAILED_DOWN), {"status": "success"}]
  )
  assert result == {"status": "success"}
  assert sleeps == [10, 20]


def test_the_wait_for_an_unavailable_eval_server_is_capped_and_never_gives_up(monkeypatch):
  outcomes = [Exception(SERVER_DOWN)] * 7 + [{"status": "success"}]
  result, sleeps = run_until_available(monkeypatch, outcomes)
  assert result == {"status": "success"}
  assert sleeps == [10, 20, 40, 80, 120, 120, 120]


def test_a_kernel_time_limit_is_a_result_and_is_raised_at_once(monkeypatch):
  with pytest.raises(eval_client.TaskTimeoutError):
    run_until_available(monkeypatch, [eval_client.TaskTimeoutError("Task abc ended with status timeout")])


def test_another_eval_server_error_is_raised_after_three_more_tries(monkeypatch):
  with pytest.raises(Exception, match="Status: 500"):
    run_until_available(monkeypatch, [Exception("Failed to submit task. Status: 500, Error: boom")] * 4)


TAMPERING = 'import sys\nsys.modules["base_kernel"].HARNESS_TIMER = "wallclock"\n'


class NeverPosts:
  def post(self, *args, **kwargs):
    raise AssertionError("the candidate was sent to the eval server")


@pytest.mark.parametrize("payload", [
  {"eval_type": "unified_test", "code": "test", "timeout": 1,
   "dependencies": {"base_kernel.py": "def kernel(x):\n    return x\n", "optimized_kernel.py": TAMPERING}},
  {"eval_type": "compilation_test", "code": TAMPERING, "timeout": 1},
  {"eval_type": "autotune", "code_template": "template", "search_space": {}, "timeout": 1, "total_timeout": 1,
   "dependencies": {"optimized_kernel.py": TAMPERING}},
])
def test_a_tampering_candidate_is_never_sent_and_comes_back_as_a_failed_result(payload):
  result = asyncio.run(eval_client.call_eval_server_until_available(NeverPosts(), "http://localhost:1", payload))
  assert result["exit_code"] == 1 and result["output"] == ""
  assert result["error"].startswith("harness tampering: ")
