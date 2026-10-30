"""Standalone tool for auto-tuning Pallas kernels using grid search on remote servers."""

import json
import logging
from typing import Any

import aiohttp

from auto_agent.client_utils.eval_client import call_eval_server_until_available
from auto_agent.constants import EVAL_SERVER_PORT, REQUEST_TIMEOUT


def sweep_candidate(code_template: str, search_space: dict) -> str:
  """The code the sweep runs for its first configuration: the template with each {name} replaced by the first
  value of its search space, as the TPU server instantiates a configuration. The tampering check reads this,
  because the best configuration of the same template becomes the node's kernel without another test."""
  code = code_template
  for name, values in search_space.items():
    if values:
      code = code.replace(f"{{{name}}}", str(values[0]))
  return code

AUTOTUNE_INDIVIDUAL_TIMEOUT = 300
# Normal autotune runs take 35-443 s, so 900 s leaves room for a slow search
# and ends a job whose TPU server never answers.
AUTOTUNE_TOTAL_TIMEOUT = 900
AUTOTUNE_POLL_INTERVAL = 30


async def autotune_kernel(
  kernel_name: str,
  code_template: str,
  search_space: dict[str, list[Any]],
  backend: str = None,
  server_addr: str = "http://localhost",
  dependencies: dict[str, str] = None,
) -> dict:
  """Runs a grid search to auto-tune a Pallas kernel on a remote server.

  Args:
      kernel_name: Name of the kernel.
      code_template: Python code containing placeholders for parameters to be
        tuned. It should produce a line like "RESULT_TIME: <float>" in its
        output to indicate performance.
      search_space: A dictionary mapping placeholder names to lists of feasible
        values.
      backend: 'tpu' or 'cpu'.
      server_addr: Address of the server (default: http://localhost).
      dependencies: A dictionary mapping dependency filenames to their contents.
        Used to provide reference code for correctness check.

  Returns:
      A dictionary containing the status, optimal parameters, and a summary of
      results.
  """
  logging.info(
    f"Starting remote autotuning for kernel: {kernel_name} on {backend}"
  )

  url = f"{server_addr}:{EVAL_SERVER_PORT}/evaluate"

  try:
    client_timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT + 10)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
      payload = {
        "eval_type": "autotune",
        "code_template": code_template,
        "search_space": search_space,
        "timeout": AUTOTUNE_INDIVIDUAL_TIMEOUT,
        "backend_type": backend,
        "total_timeout": AUTOTUNE_TOTAL_TIMEOUT,
        "dependencies": dependencies,
      }
      # An unavailable eval server is waited out, so the sweep records only a real result.
      result = await call_eval_server_until_available(
        session,
        f"{server_addr}:{EVAL_SERVER_PORT}",
        payload,
        kernel_name,
        poll_interval=AUTOTUNE_POLL_INTERVAL,
        client_wait_timeout=REQUEST_TIMEOUT,
      )

      if result["exit_code"] == 0:
        try:
          output_data = json.loads(result["output"])
          logging.info(
            f"Autotuning completed. Best config: {output_data['best_cfg']}"
            f" with time {output_data['best_time']} ms and "
            f" speedup {output_data.get('best_speedup')}"
          )
          return {
            "status": "success",
            "message": "Autotuning completed",
            "best_config": output_data["best_cfg"],
            "best_time_ms": output_data["best_time"],
            "best_speedup": output_data.get("best_speedup"),
            "best_device_kind": output_data.get("best_device_kind"),
            "best_output": output_data["best_output"],
            "all_results": output_data.get("all_results", []),
          }
        except json.JSONDecodeError:
          logging.warning("Failed to decode JSON from server output.")
          return {
            "status": "success",
            "message": "Autotuning completed (raw output)",
            "raw_output": result["output"],
          }
      else:
        try:
          output_data = json.loads(result["output"])
          return {
            "status": "failed",
            "message": result["error"] or "Autotune failed on server",
            "all_results": output_data.get("all_results", []),
          }
        except Exception:
          return {
            "status": "failed",
            "message": result["error"] or "Autotune failed on server",
            "server_output": result["output"],
          }

  except aiohttp.ClientConnectorError:
    return {
      "status": "error",
      "message": (
        f"Could not connect to server at {url}. Make sure it is running."
      ),
    }

  except Exception as e:
    return {"status": "error", "message": str(e)}
