"""The saved session.json of a search node."""
import json
import logging
import os
import time

logger = logging.getLogger(__name__)


def load_session(session_dir: str):
  """The saved session of the node folder, or None when it has none or its session.json does not parse: a copy
  cut during an upload moves to <run>/rerun_archive/<UTC time>/<node>/session.json, so the node starts again
  instead of failing its task."""
  path = os.path.join(session_dir, "session.json")
  if not os.path.exists(path):
    return None
  with open(path) as f:
    try:
      return json.load(f)
    except ValueError as error:
      node_dir = os.path.normpath(session_dir)
      archive = os.path.join(os.path.dirname(os.path.dirname(node_dir)), "rerun_archive",
                             time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()), os.path.basename(node_dir))
      os.makedirs(archive, exist_ok=True)
      os.replace(path, os.path.join(archive, "session.json"))
      logger.error("%s does not parse (%s); moved to %s, the node starts again", path, error, archive)
      return None
