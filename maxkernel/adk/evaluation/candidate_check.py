"""A static check of a candidate kernel for code that reaches past the kernel into the harness around it."""

import ast

# Modules of the search and zero-shot harnesses, which a candidate has no reason to import.
HARNESS_MODULES = ("test_harness", "harness_code", "evaluation", "auto_agent")
# The reference module and its timer setting, which only the harness reads.
HARNESS_NAMES = ("base_kernel", "HARNESS_TIMER")


def harness_tampering(source: str) -> str | None:
  """What in a candidate's source reaches into the harness, or None: sys.modules, the name or string base_kernel
  or HARNESS_TIMER, a setattr() call, an attribute assignment on an imported module (time.perf_counter = ...),
  or an import of a harness module. Source that does not parse is left to the compile step."""
  try:
    tree = ast.parse(source)
  except SyntaxError:
    return None
  imported = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for alias in node.names:
        imported.add((alias.asname or alias.name).split(".")[0])
        if alias.name.split(".")[0] in HARNESS_MODULES + HARNESS_NAMES:
          return f"import {alias.name} at line {node.lineno}"
    elif isinstance(node, ast.ImportFrom):
      if (node.module or "").split(".")[0] in HARNESS_MODULES + HARNESS_NAMES:
        return f"from {node.module} import at line {node.lineno}"
      imported.update(alias.asname or alias.name for alias in node.names)
  for node in ast.walk(tree):
    if isinstance(node, ast.Attribute):
      if isinstance(node.value, ast.Name) and node.value.id == "sys" and node.attr == "modules":
        return f"sys.modules at line {node.lineno}"
      if node.attr in HARNESS_NAMES:
        return f"{ast.unparse(node)} at line {node.lineno}"
    elif isinstance(node, ast.Name) and node.id in HARNESS_NAMES:
      return f"{node.id} at line {node.lineno}"
    elif isinstance(node, ast.Constant) and node.value in HARNESS_NAMES:
      return f"the string {node.value!r} at line {node.lineno}"
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "setattr":
      return f"setattr() at line {node.lineno}"
    elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
      targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
      for target in targets:
        root = target
        while isinstance(root, (ast.Attribute, ast.Subscript)):
          root = root.value
        if isinstance(target, ast.Attribute) and isinstance(root, ast.Name) and root.id in imported:
          return f"assignment to {ast.unparse(target)} at line {node.lineno}"
  return None
