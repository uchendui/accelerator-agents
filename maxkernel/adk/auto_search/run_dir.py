"""The run dir of one problem's search under $WORKDIR/search_runs."""

import glob
import os
import time


def run_dir(workdir, problem_id, algorithm):
  """Returns the latest existing "<problem_id>_run_<algorithm>_<time>" dir
  under workdir/search_runs, so a relaunch resumes from its search graph,
  or a new one named with the current time."""
  existing = sorted(
    glob.glob(
      os.path.join(workdir, "search_runs", f"{problem_id}_run_{algorithm}_*")
    )
  )
  if existing:
    return existing[-1]
  timestamp = time.strftime("%Y%m%d_%H%M%S")
  return os.path.join(
    workdir, "search_runs", f"{problem_id}_run_{algorithm}_{timestamp}"
  )
