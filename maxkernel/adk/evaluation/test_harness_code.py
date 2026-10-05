import ast

import pytest

from evaluation.harness_code import HARNESS_TEMPLATE


class Array:
  shape = (1,)

  def __init__(self, value):
    self.value = value


class Jnp:
  @staticmethod
  def allclose(expected, actual, atol, rtol):
    return abs(expected.value - actual.value) <= atol + rtol * abs(expected.value)


def _outputs_match():
  tree = ast.parse(HARNESS_TEMPLATE.template)
  function = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "outputs_match"
  )
  namespace = {"jnp": Jnp}
  exec(compile(ast.Module(body=[function], type_ignores=[]), "<harness>", "exec"), namespace)
  return namespace["outputs_match"]


def test_outputs_match_uses_one_tolerance_per_output():
  matches = _outputs_match()
  expected = [Array(1.0), Array(2.0)]

  assert matches(expected, [Array(1.02), Array(2.0)], [0.03, 0.0], [0.0, 0.0])
  assert not matches(expected, [Array(1.02), Array(2.01)], [0.03, 0.0], [0.0, 0.0])
  with pytest.raises(ValueError):
    matches(expected, expected, [0.03], [0.0, 0.0])
