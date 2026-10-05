from dataclasses import dataclass
from typing import List, Optional, Union


def normalize_tolerance(value):
  if isinstance(value, list):
    return [float(item) for item in value]
  return float(value)


@dataclass
class KernelTask:
  task_id: str
  description: Optional[str] = None
  input_gen_code: Optional[str] = None
  atol: Optional[Union[float, List[float]]] = None
  rtol: Optional[Union[float, List[float]]] = None
