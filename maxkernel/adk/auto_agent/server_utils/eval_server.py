import asyncio
import atexit
import json
import logging
import os
import re
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

import aiohttp
import requests
import yaml
from dotenv import find_dotenv, load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from auto_agent.constants import EVAL_SERVER_PORT, REQUEST_TIMEOUT

# TPU_VERSION comes from the exported environment or from the adk/.env file, like the other MaxKernel processes.
load_dotenv(find_dotenv(usecwd=True))
# jax device kinds of the TPU generation this hub evaluates for, keyed by the TPU_VERSION the launchers set
DEVICE_KINDS = {"TPU v5e": "TPU v5 lite", "TPU v6e": "TPU v6 lite", "TPU 7x": "TPU7x"}
EXPECTED_DEVICE_KIND = DEVICE_KINDS[os.environ["TPU_VERSION"]]

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
  # Start background task to clean up old completed tasks (TTL: 30 mins / 1800 seconds)
  cleanup_task = asyncio.create_task(
    clean_old_tasks_periodically(ttl_seconds=1800, check_interval_seconds=60)
  )
  reload_task = asyncio.create_task(reload_backends_periodically())
  yield
  # Clean up background tasks on shutdown
  for task in (cleanup_task, reload_task):
    task.cancel()
  for task in (cleanup_task, reload_task):
    try:
      await task
    except asyncio.CancelledError:
      pass
  logging.info("Shutting down eval_server, cleaning up tunnels...")
  evaluator._cleanup_tunnels()


async def clean_old_tasks_periodically(
  ttl_seconds: int = 1800, check_interval_seconds: int = 60
):
  """Periodically removes completed tasks older than ttl_seconds."""
  while True:
    try:
      await asyncio.sleep(check_interval_seconds)
      now = time.time()
      to_remove = []
      for task_id, task_info in list(_tasks.items()):
        # Only clean up tasks that have finished (are not queued or running)
        if task_info["status"] not in [TaskStatus.QUEUED, TaskStatus.RUNNING]:
          completed_at = task_info.get("completed_at")
          if completed_at and (now - completed_at) > ttl_seconds:
            to_remove.append(task_id)

      for task_id in to_remove:
        _tasks.pop(task_id, None)
        logging.info(f"Cleaned up expired task {task_id} from memory.")
    except asyncio.CancelledError:
      break
    except Exception as e:
      logging.error(f"Error in task cleanup loop: {e}")


app = FastAPI(
  title="Agent Evaluation Server", version="1.0.0", lifespan=lifespan
)

backend_semaphore = asyncio.Semaphore(1)
# Seconds a backend gets beyond the job's own timeout to send its answer.
BACKEND_GRACE = 10
# Seconds between re-reads of the config, which the GKE hub rewrites as TPU server pods join and leave.
BACKENDS_RELOAD_SECONDS = 30


async def reload_backends_periodically(interval_seconds=BACKENDS_RELOAD_SECONDS):
  while True:
    try:
      await asyncio.sleep(interval_seconds)
      # Backend() asks a new backend for its TPU version over HTTP, so the reload runs off the loop.
      await asyncio.to_thread(evaluator.reload)
    except asyncio.CancelledError:
      break
    except Exception as e:
      logging.error(f"Error reloading backends: {e}")


class Backend:
  def __init__(self, name: str, ip: str, port: int, backend_type: str = "tpu"):
    self.name = name
    self.ip = ip
    self.port = port
    self.backend_type = backend_type  # "tpu" or "cpu"
    self.status = "available"

    # Get version info for display purposes
    if backend_type == "tpu":
      url = f"http://{self.ip}:{self.port}/get_tpu_version"
      try:
        resp = requests.post(url, timeout=10)
        if resp.status_code == 200:
          data = resp.json()
          version = data.get("tpu_version") if isinstance(data, dict) else None
          if version is None or version == "TPU version not found":
            logging.warning(f"Failed to get TPU version from {url}: {version}")
            self.version = None
            self.status = "down"
          else:
            self.version = version
        else:
          logging.warning(f"Failed to get TPU version from {url}: HTTP {resp.status_code}")
          self.version = None
          self.status = "down"
      except Exception as e:
        logging.warning(f"Failed to get TPU version from {url}: {e}")
        self.version = None
        self.status = "down"
    else:
      self.version = "CPU"

  def __str__(self):
    return f"{self.name}: {self.version} ({self.ip}:{self.port})"

  def __repr__(self):
    return self.__str__()

  def set_status(self, status: str):
    if status not in ["available", "busy", "down"]:
      raise ValueError("Status must be 'available', 'busy', or 'down'")
    self.status = status

  def get_status(self):
    return self.status


class EvalTypes(Enum):
  CORRECTNESS_TEST = "correctness_test"
  COMPILATION_TEST = "compilation_test"
  PERFORMANCE_TEST = "performance_test"
  UNIFIED_TEST = "unified_test"
  PROFILE = "profile"
  AUTOTUNE = "autotune"


class EvalRequest(BaseModel):
  eval_type: EvalTypes
  timeout: int
  code: Optional[str] = None
  code_template: Optional[str] = None  # For autotune
  search_space: Optional[dict] = None  # For autotune
  backend_type: Optional[str] = None  # "tpu", "cpu", or None for any available
  dependencies: Optional[dict] = None
  total_timeout: Optional[int] = None  # For autotune
  client_wait_timeout: Optional[int] = None


class TaskStatus(str, Enum):
  QUEUED = "queued"
  RUNNING = "running"
  SUCCESS = "success"
  FAILED = "failed"
  TIMEOUT = "timeout"


class TaskResponse(BaseModel):
  task_id: str
  status: TaskStatus
  result: Optional[dict] = None
  error: Optional[str] = None


# For evaluation task tracking. task_id -> {status, request, result_or_error}
_tasks = {}


class Evaluator:
  def __init__(self, cfg_path="eval_config.yaml"):
    self.cfg_path = cfg_path
    with open(cfg_path, "r") as file:
      self.config = yaml.safe_load(file)

    self.backends = []

    self.tunnels = []
    self._load_backends()

    logging.info(f"Evaluator initialized with backends: {self.backends}")

  def _load_backends(self):
    if "backends" in self.config and self.config["backends"]:
      logging.info("Using 'backends' configuration format")
      for backend_config in self.config["backends"]:
        # Check if gcloud TPU VM tunnel is requested
        self._create_tunnel(backend_config)

        backend_obj = Backend(
          name=backend_config["name"],
          ip=backend_config["ip"],
          port=backend_config["port"],
          backend_type=backend_config.get("type", "tpu"),
        )
        self.backends.append(backend_obj)

      # Register cleanup
      if self.tunnels:
        logging.info(f"Registering cleanup for {len(self.tunnels)} tunnels.")
        atexit.register(self._cleanup_tunnels)

    else:
      raise ValueError(
        "No backends configured in eval_config.yaml. Please use the 'backends' format."
      )

  def reload(self):
    """Applies the config's current backend list: a loaded backend keeps its object and status, a new one is
    created with its tunnel, and one that left the list is dropped (a task running on it still finishes)."""
    with open(self.cfg_path, "r") as file:
      config = yaml.safe_load(file) or {}
    loaded = {
      (backend.name, backend.ip, backend.port, backend.backend_type): backend
      for backend in self.backends
    }
    backends = []
    for backend_config in config.get("backends") or []:
      key = (
        backend_config["name"], backend_config["ip"], backend_config["port"],
        backend_config.get("type", "tpu"),
      )
      backend = loaded.pop(key, None)
      if backend is None:
        self._create_tunnel(backend_config)
        backend = Backend(*key)
        logging.info(f"Added backend {backend.name}")
      elif backend.version is None:  # its server did not answer when it was added, so ask again
        backend = Backend(*key)
      backends.append(backend)
    for backend in loaded.values():
      logging.info(f"Removed backend {backend.name}")
    self.backends = backends
    self.config = config

  def _create_tunnel(self, backend_config):
    if "tpu_vm" in backend_config:
      local_port = backend_config["port"]
      gt = backend_config["tpu_vm"]
      tpu_name = gt["tpu_name"]
      zone = gt["zone"]
      project = gt["project"]
      remote_port = gt["port"]

      logging.info(f"Constructing gcloud SSH tunnel to {tpu_name}...")

      cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        tpu_name,
        f"--zone={zone}",
        f"--project={project}",
        "--",
        "-N",
        "-L",
        f"{local_port}:localhost:{remote_port}",
      ]

      try:
        process = subprocess.Popen(cmd)
        self.tunnels.append(process)
        logging.info(
          f"Gcloud SSH tunnel started for {backend_config['name']} (PID: {process.pid})"
        )
        time.sleep(10)  # Blocking sleep as requested
      except Exception as e:
        logging.error(f"Failed to start gcloud SSH tunnel: {e}")

  def _cleanup_tunnels(self):
    if self.tunnels:
      logging.info("Cleaning up SSH tunnels...")
      for p in self.tunnels:
        try:
          p.terminate()
          p.wait(timeout=5)
          logging.info(f"Tunnel PID {p.pid} terminated.")
        except Exception as e:
          logging.error(f"Failed to terminate tunnel PID {p.pid}: {e}")
          try:
            p.kill()
          except:
            pass

  def get_available_backend(self, backend_type: Optional[str] = None):
    """
    Get an available backend without blocking, optionally filtered by type.
    """
    for backend in self.backends:
      if backend.get_status() == "available":
        if backend_type is None or backend.backend_type == backend_type:
          return backend
    return None


evaluator = Evaluator()


@app.get("/health")
async def health_check():
  return {"status": "healthy"}


# Asynchronous task polling architecture for evaluation.
@app.post("/evaluate", status_code=202)
async def evaluate(request: EvalRequest, background_tasks: BackgroundTasks):
  task_id = str(uuid.uuid4())
  _tasks[task_id] = {"status": TaskStatus.QUEUED, "request": request}
  background_tasks.add_task(run_evaluation_task, task_id, request)
  return {"task_id": task_id, "status": TaskStatus.QUEUED}


@app.get("/status/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
  if task_id not in _tasks:
    raise HTTPException(status_code=404, detail="Task not found")
  task_info = _tasks[task_id]
  return {
    "task_id": task_id,
    "status": task_info["status"],
    "result": task_info.get("result"),
    "error": task_info.get("error"),
  }


async def run_evaluation_task(task_id: str, request: EvalRequest):
  try:
    result = await _perform_evaluation(task_id, request)
    _tasks[task_id]["status"] = TaskStatus.SUCCESS
    _tasks[task_id]["result"] = result
  except HTTPException as e:
    if e.status_code == 408:
      _tasks[task_id]["status"] = TaskStatus.TIMEOUT
    else:
      _tasks[task_id]["status"] = TaskStatus.FAILED
    _tasks[task_id]["error"] = str(e.detail)
  except Exception as e:
    _tasks[task_id]["status"] = TaskStatus.FAILED
    _tasks[task_id]["error"] = str(e)
  finally:
    _tasks[task_id]["completed_at"] = time.time()


async def _perform_evaluation(task_id: str, request: EvalRequest):
  if request.eval_type not in EvalTypes:
    raise HTTPException(status_code=400, detail="Invalid evaluation type")

  start_queue_time = time.time()
  max_queue_time = (
    request.client_wait_timeout
    if request.client_wait_timeout is not None
    else REQUEST_TIMEOUT
  )

  while True:
    while True:
      matching_backends = [
        backend
        for backend in evaluator.backends
        if request.backend_type is None
        or backend.backend_type == request.backend_type
      ]
      if matching_backends and all(
        backend.get_status() == "down" for backend in matching_backends
      ):
        backend_type = request.backend_type or "configured"
        raise HTTPException(
          status_code=503,
          detail=f"Every TPU server of type '{backend_type}' is down",
        )
      if time.time() - start_queue_time > max_queue_time:
        raise HTTPException(
          status_code=408,
          detail="Timed out waiting for an available backend in queue",
        )

      async with backend_semaphore:
        backend = evaluator.get_available_backend(
          backend_type=request.backend_type
        )
        if backend:
          backend.set_status("busy")
          _tasks[task_id]["status"] = TaskStatus.RUNNING
          backend_ip = backend.ip
          backend_port = backend.port
          backend_name = backend.name
          backend_type = backend.backend_type
          break
      await asyncio.sleep(0.1)

    try:
      requested_type_msg = (
        f" (requested: {request.backend_type})" if request.backend_type else ""
      )
      logging.info(
        f"Starting evaluation on {backend_type} backend '{backend_name}' "
        f"({backend_ip}:{backend_port}) for {request.eval_type.value}"
        f"{requested_type_msg}"
      )

      payload = {
        "eval_type": request.eval_type.value,
        "timeout": request.timeout,
      }
      if request.eval_type == EvalTypes.AUTOTUNE:
        payload["code_template"] = request.code_template
        payload["search_space"] = request.search_space
        backend_timeout = request.total_timeout
        payload["total_timeout"] = request.total_timeout
        payload["dependencies"] = request.dependencies
      else:
        payload["code"] = request.code
        payload["dependencies"] = request.dependencies
        backend_timeout = request.timeout

      client_timeout = aiohttp.ClientTimeout(
        total=backend_timeout + BACKEND_GRACE, sock_connect=30
      )
      async with aiohttp.ClientSession(timeout=client_timeout) as session:
        if backend_type == "tpu":
          try:
            async with session.post(
              f"http://{backend_ip}:{backend_port}/get_tpu_version",
              timeout=aiohttp.ClientTimeout(total=10),
            ) as version_response:
              data = await version_response.json()
              version = data["tpu_version"] if version_response.status == 200 and isinstance(data, dict) else None
          except Exception:
            version = None
          if version != backend.version:
            logging.error(
              f"TPU version mismatch for backend {backend_name}: recorded {backend.version}, got {version}"
            )
            backend.set_status("down")
            _tasks[task_id]["status"] = TaskStatus.QUEUED
            continue
        async with session.post(
          f"http://{backend_ip}:{backend_port}/{request.eval_type.value}",
          json=payload,
        ) as response:
          result = await response.json()
          logging.info(
            f"Received response from {backend_name}: {response.status}"
          )

          if response.status != 200:
            if response.status == 408:
              raise HTTPException(
                status_code=408,
                detail=f"Backend evaluation timed out. Timeout was set to {backend_timeout} seconds.",
              )
            raise HTTPException(
              status_code=response.status,
              detail=result.get("detail", "Backend evaluation failed"),
            )

      # A timing without a chip line is a harness error. A wrong real kind marks the backend down.
      kinds = set()
      if request.eval_type == EvalTypes.AUTOTUNE:
        if result["exit_code"] == 0:
          kinds = {
            entry.get("device_kind")
            for entry in json.loads(result["output"])["all_results"]
            if entry.get("status") == "success"
          }
      elif re.search(r"RESULT_TIME:", result["output"] or ""):
        kind_match = re.search(r"DEVICE_KIND:\s*(.+)", result["output"])
        kinds = {kind_match.group(1).strip() if kind_match else None}
      if None in kinds:
        raise HTTPException(
          status_code=500,
          detail="the result carries a timing without a DEVICE_KIND line",
        )
      if kinds and kinds != {EXPECTED_DEVICE_KIND}:
        logging.error(
          f"Device kind mismatch for backend {backend_name}: expected {EXPECTED_DEVICE_KIND}, got {kinds}"
        )
        backend.set_status("down")
        _tasks[task_id]["status"] = TaskStatus.QUEUED
        continue

      logging.info(
        f"Evaluation completed on {backend_name}, marking as available"
      )
      return result

    except aiohttp.ClientConnectionError as e:
      logging.error(f"Connection to backend {backend_name} failed: {e}")
      backend.set_status("down")
      _tasks[task_id]["status"] = TaskStatus.QUEUED
    except HTTPException as e:
      logging.info(f"HTTPException occurred during evaluation: {e.detail}")
      raise e
    except Exception as e:
      logging.error(f"Error occurred while evaluating on {backend_name}: {e}")
      raise HTTPException(
        status_code=500,
        detail=f"Backend evaluation failed: {type(e).__name__}: {e}",
      )
    finally:
      if backend.get_status() == "busy":
        backend.set_status("available")


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=EVAL_SERVER_PORT)
