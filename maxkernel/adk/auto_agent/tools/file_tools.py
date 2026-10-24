"""File-related tools for subagents."""

import contextlib
import contextvars
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from google.adk.tools import FunctionTool, ToolContext
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from mcp import StdioServerParameters

from auto_agent.config import WORKDIR
from auto_agent.tools.test_harness import TEST_TEMPLATE

# The harness writes these into the node's workdir. session.json holds the full
# event history, so reading it grows the prompt past the model input limit.
HARNESS_FILES = ("session.json", "token_metrics.json", "timing_metrics.json")

# The prompts name these node files as readable.
READABLE_STATE_KEYS = (
  "base_kernel_path",
  "optimized_kernel_path",
  "kernel_plan_path",
  "test_file_path",
  "profiling_script_path",
  "autotune_specs_path",
  "autotune_results_path",
)

# Installed once per host with:
#   npm install --prefix ~/.local/share/mcp-server-filesystem \
#     @modelcontextprotocol/server-filesystem@2026.8.31
# An `npx -y` start resolves the package through npm on every start, which
# timed out with hundreds of searches on one host.
SERVER_FILESYSTEM_BIN = os.path.expanduser(
  "~/.local/share/mcp-server-filesystem/node_modules/.bin/mcp-server-filesystem"
)
if not os.path.isfile(SERVER_FILESYSTEM_BIN):
  raise FileNotFoundError(
    f"{SERVER_FILESYSTEM_BIN} is missing; install it with: npm install --prefix"
    " ~/.local/share/mcp-server-filesystem"
    " @modelcontextprotocol/server-filesystem@2026.8.31"
  )

# The MCPToolsets of the running search (set by search_toolsets); outside a
# search (the HITL agent) one process-wide dict is used instead, as before.
_search = contextvars.ContextVar("search_toolsets")
_shared_toolsets = {}


def _toolsets():
  toolsets = _search.get(None)
  return _shared_toolsets if toolsets is None else toolsets


class NodeFilesystemTool(BaseTool):
  """Restricts an MCP filesystem tool to one search node's files."""

  def __init__(self, inner):
    super().__init__(name=inner.name, description=inner.description)
    self._inner = inner

  def _get_declaration(self):
    return self._inner._get_declaration()

  async def run_async(self, *, args, tool_context):
    state = tool_context.state
    workdir = state.get("workdir")
    root = Path(workdir or WORKDIR).resolve()

    def access_denied(text):
      return {
        "content": [{"type": "text", "text": text}],
        "isError": True,
      }

    path = args.get("path") or ""
    target = Path(os.path.expanduser(path)).resolve()
    if not path.strip() or not target.is_relative_to(root):
      return access_denied(
        "Access denied because the path is outside the node's folder "
        f"{root}."
      )
    if target.name in HARNESS_FILES:
      return access_denied("Access denied because the path is a harness file.")

    allowed = None
    if workdir:
      allowed = {
        Path(state[key]).resolve()
        for key in READABLE_STATE_KEYS
        if state.get(key)
      }
      if self.name == "read_file" and target not in allowed:
        return access_denied("Access denied because the path is not readable.")
      if self.name == "list_directory" and target != root:
        return access_denied(
          "Access denied because the path is not the node folder."
        )

    result = await self._inner.run_async(args=args, tool_context=tool_context)
    if self.name == "list_directory":
      harness_entries = {f"[FILE] {name}" for name in HARNESS_FILES}

      def filter_directory_listing(text):
        if allowed is None:
          return "\n".join(
            line for line in text.splitlines() if line not in harness_entries
          )
        return "\n".join(
          line
          for line in text.splitlines()
          if line.startswith("[FILE] ")
          and (target / line.removeprefix("[FILE] ")).resolve() in allowed
        )

      for content in result.get("content", []):
        if content.get("type") == "text":
          content["text"] = filter_directory_listing(content["text"])
    # The MCP server returns the text twice, as content and as structuredContent; the model gets it once.
    return {key: result[key] for key in ("content", "isError") if key in result}


class SearchToolset(BaseToolset):
  """A filesystem toolset shared by name, backed by one MCP server per search.

  The agents are module-level objects built at import, so they keep one toolset
  object each; each search gets its own server pair through search_toolsets(),
  because one shared pair could not serve a thousand concurrent problems (its
  list_tools calls timed out).
  """

  def __init__(self, tool_filter):
    super().__init__(tool_filter=tool_filter)

  def _current(self):
    toolsets = _toolsets()
    toolset = toolsets.get(self)
    if toolset is None:
      toolset = MCPToolset(
        connection_params=StdioConnectionParams(
          server_params=StdioServerParameters(
            command=SERVER_FILESYSTEM_BIN,
            args=[os.path.abspath(WORKDIR)],
          ),
          timeout=60,
        ),
        tool_filter=self.tool_filter,
      )
      toolsets[self] = toolset
    return toolset

  async def get_tools(self, readonly_context=None):
    tools = await self._current().get_tools(readonly_context)
    return [NodeFilesystemTool(tool) for tool in tools]

  async def close(self):
    toolset = _toolsets().pop(self, None)
    if toolset is not None:
      await toolset.close()


@contextlib.asynccontextmanager
async def search_toolsets():
  """Runs one search with its own filesystem server pair, closed at its end."""
  token = _search.set({})
  try:
    yield
  finally:
    toolsets = _search.get()
    _search.reset(token)
    for toolset in toolsets.values():
      await toolset.close()


filesystem_tool_r = SearchToolset(["list_directory", "read_file"])
filesystem_tool_rw = SearchToolset(["list_directory", "read_file", "write_file"])


def restricted_write_file(state_key: str, description: str) -> FunctionTool:
  """Creates a tool that writes content to a path stored in session state.

  Args:
      state_key: The key in tool_context.state that holds the target file path.
      description: The description of the tool for the agent.
  """

  def _write_file(content: str, tool_context: ToolContext) -> str:
    target_path = tool_context.state.get(state_key)
    if not target_path:
      return f"Error: Path variable '{state_key}' not found in session state."

    base = Path(WORKDIR).resolve()
    target = Path(target_path).resolve()

    try:
      if not target.is_relative_to(base):
        return f"Error: Access denied. Path is outside {WORKDIR}"
    except ValueError:
      return "Error: Invalid path or access denied."

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Successfully wrote to {target}"

  _write_file.__name__ = "restricted_write_file"
  _write_file.__doc__ = description
  return FunctionTool(_write_file)


def write_autotune_specs_tool_fn(
  kernel_name: str,
  code_template: str,
  search_space: Dict[str, List[Any]],
  tool_context: ToolContext,
) -> str:
  """Writes the structured autotuning specifications to autotune_specs_path in session state.

  Args:
      kernel_name: The name of the Pallas kernel.
      code_template: The kernel source code template with placeholders like {BLOCK_M}.
      search_space: Dictionary mapping placeholder names to lists of suggested tuning values.
  """
  target_path = tool_context.state.get("autotune_specs_path")
  if not target_path:
    return (
      "Error: Path variable 'autotune_specs_path' not found in session state."
    )

  base = Path(WORKDIR).resolve()
  target = Path(target_path).resolve()

  try:
    if not target.is_relative_to(base):
      return f"Error: Access denied. Path is outside {WORKDIR}"
  except ValueError:
    return "Error: Invalid path or access denied."

  content_dict = {
    "kernel_name": kernel_name,
    "code_template": code_template,
    "search_space": search_space,
  }

  target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text(json.dumps(content_dict, indent=2))
  return f"Successfully wrote structured autotuning specs to {target}"


def write_test_file_tool_fn(
  content: str,
  kernel_name: str,
  tool_context: ToolContext,
  atol: float | list[float] = 1e-2,
  rtol: float | list[float] = 1e-2,
  sort_before_compare: bool = False,
) -> str:
  """Writes the generated input generation snippet to the test file using the rigorous harness template.

  Args:
      content: The Python code snippet containing the `get_inputs()` function.
      kernel_name: The exact function name of the base kernel entry point (e.g., "computation").
      atol: Absolute tolerance for numerical correctness checks. Default 1e-2. Set lower for F32.
      rtol: Relative tolerance for numerical correctness checks. Default 1e-2. Set lower for F32.
      sort_before_compare: Sort each output leaf along its last axis before comparison.
  """
  target_path = tool_context.state.get("test_file_path")
  if not target_path:
    return "Error: Path variable 'test_file_path' not found in session state."

  base = Path(WORKDIR).resolve()
  target = Path(target_path).resolve()

  try:
    if not target.is_relative_to(base):
      return f"Error: Access denied. Path is outside {WORKDIR}"
  except ValueError:
    return "Error: Invalid path or access denied."

  target.parent.mkdir(parents=True, exist_ok=True)

  import re

  content = re.sub(r"^```(python)?\n", "", content.strip())
  content = re.sub(r"\n```$", "", content)

  if not isinstance(sort_before_compare, bool):
    raise TypeError("sort_before_compare must be a boolean")

  # Save kernel_name and comparison settings to state for downstream agents
  tool_context.state["kernel_name"] = kernel_name
  tool_context.state["atol"] = atol
  tool_context.state["rtol"] = rtol
  tool_context.state["sort_before_compare"] = sort_before_compare

  full_content = TEST_TEMPLATE.format(
    input_gen_code=content,
    atol=atol,
    rtol=rtol,
    sort_before_compare=sort_before_compare,
    kernel_name=kernel_name,
  )

  target.write_text(full_content)
  return f"Successfully wrote to {target}"


write_test_file_tool_fn.__name__ = "restricted_write_file"
write_test_file_tool = FunctionTool(write_test_file_tool_fn)
write_optimized_kernel_tool = restricted_write_file(
  "optimized_kernel_path", "Writes the optimized Pallas kernel file."
)
write_optimization_plan_tool = restricted_write_file(
  "kernel_plan_path", "Writes the optimization plan."
)
write_profiling_script_tool = restricted_write_file(
  "profiling_script_path", "Writes the profiling script."
)
write_autotune_specs_tool_fn.__name__ = "restricted_write_file"
write_autotune_specs_tool = FunctionTool(write_autotune_specs_tool_fn)
write_base_kernel_tool = restricted_write_file(
  "base_kernel_path", "Writes the base kernel file."
)

__all__ = [
  "filesystem_tool_r",
  "filesystem_tool_rw",
  "search_toolsets",
  "write_test_file_tool",
  "write_optimized_kernel_tool",
  "write_optimization_plan_tool",
  "write_profiling_script_tool",
  "write_autotune_specs_tool",
  "write_base_kernel_tool",
]
