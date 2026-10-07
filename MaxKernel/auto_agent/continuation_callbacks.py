from collections.abc import Callable
import inspect
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

_CONTINUE_PROMPT = (
  "Continue exactly from where your answer stopped. Do not repeat or restart anything."
)
_CUT_FINISH_REASONS = {
  types.FinishReason.MAX_TOKENS,
  types.FinishReason.CONTINUATION,
}
_MAX_CONTINUE_REQUESTS = 3


def create_continuation_callbacks(
  agent: LlmAgent,
) -> tuple[Callable, Callable]:
  requests: dict[str, LlmRequest] = {}

  async def before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
  ) -> None:
    requests[callback_context.invocation_id] = llm_request.model_copy(
      update={"contents": list(llm_request.contents)}
    )

  async def after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
  ) -> Optional[LlmResponse]:
    original_request = requests.pop(callback_context.invocation_id)
    if llm_response.finish_reason not in _CUT_FINISH_REASONS:
      return None

    response = llm_response
    merged_parts = []
    for continue_index in range(_MAX_CONTINUE_REQUESTS + 1):
      if response.finish_reason in _CUT_FINISH_REASONS and (
        not response.content or not response.content.parts
      ):
        raise RuntimeError(
          f"{agent.name} returned a cut response without content parts"
        )
      if response.content and response.content.parts:
        merged_parts.extend(response.content.parts)
      if response.finish_reason not in _CUT_FINISH_REASONS:
        return response.model_copy(
          update={
            "content": types.Content(role="model", parts=merged_parts),
          }
        )
      if continue_index == _MAX_CONTINUE_REQUESTS:
        raise RuntimeError(
          f"{agent.name} response remained cut after "
          f"{_MAX_CONTINUE_REQUESTS} continuation requests"
        )

      continuation_request = original_request.model_copy(
        update={
          "contents": [
            *original_request.contents,
            types.Content(role="model", parts=list(merged_parts)),
            types.Content(
              role="user",
              parts=[types.Part(text=_CONTINUE_PROMPT)],
            ),
          ]
        }
      )
      # ADK resolves the agent's BaseLlm at llm_agent.py:690-710. Its
      # non-streaming call yields one response (base_llm.py:131-150).
      response = await anext(
        agent.canonical_model.generate_content_async(continuation_request)
      )

    raise AssertionError("unreachable")

  return before_model_callback, after_model_callback


def add_continuation_callbacks(agent: LlmAgent) -> None:
  before_callback, after_callback = create_continuation_callbacks(agent)
  existing_before = agent.before_model_callback
  agent.before_model_callback = [
    *(existing_before if isinstance(existing_before, list) else [existing_before]),
    before_callback,
  ] if existing_before else [before_callback]

  existing_after = agent.after_model_callback
  after_callbacks = (
    existing_after if isinstance(existing_after, list) else [existing_after]
  ) if existing_after else []

  async def composed_after_callback(callback_context, llm_response):
    merged_response = await after_callback(callback_context, llm_response)
    response = merged_response or llm_response
    for callback in after_callbacks:
      callback_response = callback(callback_context, response)
      if inspect.isawaitable(callback_response):
        callback_response = await callback_response
      if callback_response:
        return callback_response
    return merged_response

  agent.after_model_callback = [composed_after_callback]


def inject_continuation_callbacks(agent: BaseAgent) -> None:
  visited = set()

  def _inject(current_agent):
    if id(current_agent) in visited:
      return
    visited.add(id(current_agent))
    if isinstance(current_agent, LlmAgent):
      add_continuation_callbacks(current_agent)

    for attr_name in dir(current_agent):
      if attr_name.startswith("_") or attr_name == "parent_agent":
        continue
      try:
        attr_value = getattr(current_agent, attr_name, None)
        if isinstance(attr_value, BaseAgent):
          _inject(attr_value)
        elif isinstance(attr_value, list):
          for item in attr_value:
            if isinstance(item, BaseAgent):
              _inject(item)
      except Exception:
        pass

  _inject(agent)
