import ast

import pytest

from evaluation.custom_types.kernel_task import normalize_tolerance as normalize_task_tolerance
from evaluation.harness_code import HARNESS_TEMPLATE


class Array:
  shape = (1,)

  def __init__(self, value):
    self.value = value


class Jnp:
  @staticmethod
  def allclose(expected, actual, atol, rtol):
    return abs(expected.value - actual.value) <= atol + rtol * abs(expected.value)


def _harness_function(name):
  tree = ast.parse(HARNESS_TEMPLATE.template)
  function = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == name
  )
  namespace = {"jnp": Jnp}
  exec(compile(ast.Module(body=[function], type_ignores=[]), "<harness>", "exec"), namespace)
  return namespace[name]


def test_outputs_match_uses_one_tolerance_per_output():
  matches = _harness_function("outputs_match")
  expected = [Array(1.0), Array(2.0)]

  assert matches(expected, [Array(1.02), Array(2.0)], [0.03, 0.0], [0.0, 0.0])
  assert not matches(expected, [Array(1.02), Array(2.01)], [0.03, 0.0], [0.0, 0.0])
  validate = _harness_function("validate_output_tolerances")
  with pytest.raises(ValueError):
    validate(len(expected), [0.03], [0.0, 0.0])


def test_normalize_tolerance_converts_yaml_exponent_strings():
  normalize = _harness_function("normalize_tolerance")

  assert normalize("1e-05") == 1e-5
  assert normalize(["1e-05", 0.0]) == [1e-5, 0.0]
  assert normalize_task_tolerance(["1e-05", 0.0]) == [1e-5, 0.0]
