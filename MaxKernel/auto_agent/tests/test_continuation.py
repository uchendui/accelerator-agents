import asyncio
from types import SimpleNamespace

import pytest
from google.adk.agents import LlmAgent
from google.adk.models import LlmRequest, LlmResponse
from google.adk.models.base_llm import BaseLlm
from google.genai import types

from auto_agent.continuation_callbacks import add_continuation_callbacks


class FakeModel(BaseLlm):
  responses: list[LlmResponse]
  requests: list[LlmRequest] = []

  async def generate_content_async(self, llm_request, stream=False):
    self.requests.append(llm_request)
    yield self.responses.pop(0)


def _response(finish_reason, *parts):
  return LlmResponse(
    content=types.Content(role="model", parts=list(parts)),
    finish_reason=finish_reason,
  )


async def _run_model_callbacks(agent, context, request, response):
  for callback in agent.before_model_callback or []:
    await callback(context, request)
  for callback in agent.after_model_callback or []:
    if replacement := await callback(context, response):
      return replacement
  return response


def test_continues_cut_response_and_keeps_function_call():
  function_call = types.FunctionCall(
    name="restricted_write_file", args={"path": "kernel.py"}
  )
  model = FakeModel(
    model="fake",
    responses=[
      _response(
        types.FinishReason.STOP,
        types.Part(function_call=function_call),
      )
    ],
  )
  agent = LlmAgent(name="ImplementKernelAgent", model=model)
  add_continuation_callbacks(agent)
  context = SimpleNamespace(invocation_id="invocation-1")
  request = LlmRequest(
    contents=[
      types.Content(role="user", parts=[types.Part(text="write the kernel")])
    ]
  )
  cut_response = _response(
    types.FinishReason.MAX_TOKENS,
    types.Part(text="partial answer"),
    types.Part(text="private reasoning", thought=True),
  )

  merged = asyncio.run(
    _run_model_callbacks(agent, context, request, cut_response)
  )

  assert [part.text for part in merged.content.parts[:2]] == [
    "partial answer",
    "private reasoning",
  ]
  assert merged.content.parts[2].function_call == function_call
  continuation_contents = model.requests[0].contents
  assert continuation_contents[-2].role == "model"
  assert continuation_contents[-2].parts[1].thought is True
  assert continuation_contents[-1] == types.Content(
    role="user",
    parts=[
      types.Part(
        text=(
          "Continue exactly from where your answer stopped. Do not repeat or restart anything."
        )
      )
    ],
  )


def test_raises_after_three_continuation_requests():
  cut_response = _response(
    types.FinishReason.CONTINUATION, types.Part(text="partial")
  )
  model = FakeModel(
    model="fake",
    responses=[cut_response.model_copy(deep=True) for _ in range(3)],
  )
  agent = LlmAgent(name="PlanKernelAgent", model=model)
  add_continuation_callbacks(agent)
  context = SimpleNamespace(invocation_id="invocation-2")

  with pytest.raises(RuntimeError, match="PlanKernelAgent.*3 continuation"):
    asyncio.run(
      _run_model_callbacks(agent, context, LlmRequest(), cut_response)
    )

  assert len(model.requests) == 3


def test_returns_stop_response_unchanged():
  model = FakeModel(model="fake", responses=[])
  agent = LlmAgent(name="TestingAgent", model=model)
  add_continuation_callbacks(agent)
  context = SimpleNamespace(invocation_id="invocation-3")
  response = _response(types.FinishReason.STOP, types.Part(text="done"))

  result = asyncio.run(
    _run_model_callbacks(agent, context, LlmRequest(), response)
  )

  assert result is response
  assert model.requests == []
