import asyncio
import importlib
import os
import sys
import types

from auto_agent.client_utils import eval_client

# The subagent packages import every agent, and the file tools among them need the MCP filesystem server
# binary, so the packages are entered without their __init__ and only the three eval steps are imported.
# This file sits outside those packages, so pytest does not import them either.
SUBAGENTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "subagents")
for package in ("auto_agent.subagents", "auto_agent.subagents.kernel_writing", "auto_agent.subagents.profiling",
                "auto_agent.subagents.autotuning"):
  module = types.ModuleType(package)
  module.__path__ = [os.path.join(SUBAGENTS, *package.split(".")[2:])]
  sys.modules.setdefault(package, module)
autotune_tool = importlib.import_module("auto_agent.subagents.autotuning.autotune_tool")
kernel_compilation = importlib.import_module("auto_agent.subagents.kernel_writing.kernel_compilation")
kernel_profile = importlib.import_module("auto_agent.subagents.profiling.kernel_profile")

SERVER_DOWN = "Failed to submit task. Status: 503, Error: Every TPU server of type 'tpu' is down"


def scripted_eval_server(monkeypatch, result):
  """The eval server answers 503 twice and then the result; the waits between the tries are recorded."""
  outcomes = iter([Exception(SERVER_DOWN), Exception(SERVER_DOWN), result])
  sleeps = []

  async def fake_call(session, url, payload, **kwargs):
    outcome = next(outcomes)
    if isinstance(outcome, BaseException):
      raise outcome
    return outcome

  async def fake_sleep(seconds):
    sleeps.append(seconds)

  monkeypatch.setattr(eval_client, "call_eval_server_async", fake_call)
  monkeypatch.setattr(eval_client.asyncio, "sleep", fake_sleep)
  return sleeps


def context(**state):
  return types.SimpleNamespace(session=types.SimpleNamespace(state=state))


def events_of(agent, ctx):
  async def run():
    return [event async for event in agent._run_async_impl(ctx)]

  return asyncio.run(run())


def test_the_compilation_step_waits_out_an_unavailable_eval_server(monkeypatch):
  sleeps = scripted_eval_server(monkeypatch, {"exit_code": 0, "output": "ok", "error": ""})
  agent = kernel_compilation.KernelCompilationChecker(name="compile", input_key="code", output_key="result")
  events = events_of(agent, context(code="print(1)"))
  assert sleeps == [10, 20]
  assert events[-1].actions.state_delta["result"] == "Success"


def test_the_profiling_step_waits_out_an_unavailable_eval_server(monkeypatch):
  sleeps = scripted_eval_server(monkeypatch, {"exit_code": 0, "output": '{"ratio": 0.25}', "error": ""})
  agent = kernel_profile.KernelProfiler(name="profile", input_key="profile_code", output_key="result",
                                        raise_exception_upon_success=False)
  events = events_of(agent, context(profile_code="print(1)"))
  assert sleeps == [10, 20]
  assert events[-1].actions.state_delta["result"]["DMAs_and_memory_transfers_ratio"] == 0.25


def test_the_autotune_sweep_waits_out_an_unavailable_eval_server(monkeypatch):
  output = '{"best_cfg": {"bm": 8}, "best_time": 2.0, "best_output": "ok"}'
  sleeps = scripted_eval_server(monkeypatch, {"exit_code": 0, "output": output})
  result = asyncio.run(autotune_tool.autotune_kernel("k", "code {bm}", {"bm": [8]}, backend="tpu"))
  assert sleeps == [10, 20]
  assert (result["status"], result["best_config"]) == ("success", {"bm": 8})
