import asyncio
import logging
import time

import aiohttp

from auto_agent.constants import REQUEST_TIMEOUT


# The longest wait between two tries while the eval server is unavailable.
RETRY_DELAY_CAP = 120


class TaskTimeoutError(Exception):
  pass


# Covers eval_server's answer grace plus one 30 s poll.
RUNNING_GRACE = 60


async def call_eval_server_async(
  session: aiohttp.ClientSession,
  eval_server_url: str,
  payload: dict,
  poll_interval: int = 10,
  client_wait_timeout: int = REQUEST_TIMEOUT,
) -> dict:
  """Calls the evaluation server asynchronously and polls for status.

  Args:
      session: aiohttp.ClientSession to use.
      eval_server_url: Base URL of the eval server (e.g.,
        "http://localhost:1245").
      payload: The request payload.
      poll_interval: Seconds to wait between polls.
      client_wait_timeout: Max seconds to wait for task completion.

  Returns:
      The result from the evaluation server.
  """
  # 1. Submit the task
  submit_url = f"{eval_server_url}/evaluate"
  logging.info(f"Submitting async task to {submit_url}")

  payload = payload.copy()
  payload["client_wait_timeout"] = client_wait_timeout

  async with session.post(submit_url, json=payload) as response:
    if response.status != 202:
      error_text = await response.text()
      raise Exception(
        f"Failed to submit task. Status: {response.status}, Error: {error_text}"
      )
    resp_data = await response.json()
    task_id = resp_data["task_id"]
    logging.info(f"Task submitted successfully. ID: {task_id}")

  # 2. Poll for status
  start_time = time.time()
  running_deadline = (
    payload.get("total_timeout") or payload["timeout"]
  ) + RUNNING_GRACE
  running_since = None
  status_url = f"{eval_server_url}/status/{task_id}"

  while True:
    if time.time() - start_time > client_wait_timeout:
      raise Exception(
        f"Client timed out waiting for task {task_id} after {client_wait_timeout} seconds"
      )

    async with session.get(status_url) as response:
      if response.status != 200:
        error_text = await response.text()
        raise Exception(
          f"Failed to get task status. Status: {response.status}, Error: {error_text}"
        )

      status_data = await response.json()
      status = status_data["status"]

      if status == "success":
        logging.info(f"Task {task_id} completed successfully.")
        return status_data["result"]
      elif status == "timeout":
        raise TaskTimeoutError(
          f"Task {task_id} ended with status {status}: {status_data.get('error')}"
        )
      elif status == "failed":
        raise Exception(
          f"Task {task_id} ended with status {status}: {status_data.get('error')}"
        )
      elif status == "queued":
        running_since = None
      elif status == "running":
        if running_since is None:
          running_since = time.time()
        elif time.time() - running_since > running_deadline:
          raise Exception(
            f"Task {task_id} stayed running for {running_deadline} seconds"
          )

      logging.info(
        f"Task {task_id} status: {status}. Waiting {poll_interval}s..."
      )
      await asyncio.sleep(poll_interval)


def eval_server_unavailable(error: BaseException) -> bool:
  """Whether the error means the eval server could not take the request: it was unreachable, or every TPU
  server of the type was down (a 503 answer, or the queued task failed with that detail)."""
  text = str(error)
  return (
    isinstance(error, (aiohttp.ClientConnectionError, asyncio.TimeoutError))
    or "Status: 503" in text
    or "is down" in text
  )


async def call_eval_server_until_available(
  session: aiohttp.ClientSession,
  eval_server_url: str,
  payload: dict,
  name: str = "",
  **kwargs,
) -> dict:
  """call_eval_server_async that keeps trying while the eval server is unavailable, with the wait doubling
  from 10 s up to RETRY_DELAY_CAP and no limit on the tries (the search's own timeout bounds them). A kernel
  that ran past its time limit (TaskTimeoutError) is a result and is raised at once. Any other error gets
  three more tries, 10, 20 and 40 s apart, and is then raised."""
  delay = 10
  other_delays = iter((10, 20, 40))
  while True:
    try:
      return await call_eval_server_async(session, eval_server_url, payload, **kwargs)
    except TaskTimeoutError:
      raise
    except Exception as e:
      if eval_server_unavailable(e):
        wait, delay = delay, min(delay * 2, RETRY_DELAY_CAP)
      else:
        wait = next(other_delays, None)
        if wait is None:
          raise
      logging.warning(f"[{name}] Eval server error: {e}. Retrying in {wait} seconds.")
      await asyncio.sleep(wait)
