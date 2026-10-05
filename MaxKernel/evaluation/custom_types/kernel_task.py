from dataclasses import dataclass
from typing import List, Optional, Union


def normalize_tolerance(value):
  if isinstance(value, list):
    return [float(item) for item in value]
  return float(value)


def normalize_sort_outputs(value):
  if not isinstance(value, bool):
    raise TypeError("sort_outputs must be a boolean")
  return value


@dataclass
class KernelTask:
  task_id: str
  description: Optional[str] = None
  input_gen_code: Optional[str] = None
  atol: Optional[Union[float, List[float]]] = None
  rtol: Optional[Union[float, List[float]]] = None
  sort_outputs: bool = False

  def __post_init__(self):
    self.sort_outputs = normalize_sort_outputs(self.sort_outputs)
