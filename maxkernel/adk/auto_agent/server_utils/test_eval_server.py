import importlib
import os

import pytest
import requests
import yaml


@pytest.fixture
def eval_server(tmp_path, monkeypatch):
  """Imports the eval server module from a directory holding a CPU-only config, because the module builds
  its own Evaluator from the working directory's eval_config.yaml at import."""
  monkeypatch.setenv("TPU_VERSION", "TPU v6e")
  (tmp_path / "eval_config.yaml").write_text(
    yaml.safe_dump({"backends": [{"name": "cpu-0", "ip": "localhost", "port": 5000, "type": "cpu"}]})
  )
  monkeypatch.chdir(tmp_path)
  return importlib.import_module("auto_agent.server_utils.eval_server")


class FakeResponse:
  status_code = 200

  def __init__(self, version):
    self.version = version

  def json(self):
    return {"tpu_version": self.version}


def write_config(path, name="tpu-0"):
  path.write_text(yaml.safe_dump({"backends": [{"name": name, "ip": "127.0.0.1", "port": 9000}]}))
  return str(path)


def test_a_down_tpu_backend_comes_back_when_it_answers_again(eval_server, tmp_path, monkeypatch):
  answers = {"version": None}

  def post(url, timeout):
    if answers["version"] is None:
      raise requests.ConnectionError("refused")
    return FakeResponse(answers["version"])

  monkeypatch.setattr(eval_server.requests, "post", post)
  evaluator = eval_server.Evaluator(write_config(tmp_path / "config.yaml"))
  (backend,) = evaluator.backends
  assert backend.get_status() == "down"
  assert backend.version is None

  answers["version"] = "TPU v6e"
  evaluator.reload()
  assert evaluator.backends == [backend]
  assert backend.get_status() == "available"
  assert backend.version == "TPU v6e"

  # The request path marks a backend down when its connection drops; the next reload asks it again.
  backend.set_status("down")
  evaluator.reload()
  assert evaluator.backends == [backend]
  assert backend.get_status() == "available"


def test_a_down_tpu_backend_that_answers_another_version_stays_down(eval_server, tmp_path, monkeypatch):
  answers = {"version": "TPU v6e"}
  monkeypatch.setattr(eval_server.requests, "post", lambda url, timeout: FakeResponse(answers["version"]))
  evaluator = eval_server.Evaluator(write_config(tmp_path / "config.yaml"))
  (backend,) = evaluator.backends
  assert backend.get_status() == "available"

  backend.set_status("down")
  answers["version"] = "TPU v5e"
  evaluator.reload()
  assert backend.get_status() == "down"
  assert backend.version == "TPU v6e"
