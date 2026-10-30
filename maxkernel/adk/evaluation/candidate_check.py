"""A static check of a candidate kernel for code that reaches past the kernel into the harness around it."""

import ast

# Modules a candidate has no reason to import: the search and zero-shot harnesses, and the ways to reach a
# module or a frame by a computed name.
HARNESS_MODULES = ("test_harness", "harness_code", "evaluation", "auto_agent", "importlib", "inspect", "gc",
                   "ctypes", "builtins")
# Names and attributes that only the harness or the interpreter's internals use.
HARNESS_NAMES = ("base_kernel", "HARNESS_TIMER", "__main__", "__builtins__", "builtins", "importlib", "inspect",
                 "gc", "ctypes")
# The same as strings, without "__main__": an ordinary kernel ends with `if __name__ == "__main__":`.
HARNESS_STRINGS = tuple(name for name in HARNESS_NAMES if name != "__main__")
HARNESS_ATTRIBUTES = ("modules", "_getframe", "__dict__")
# Calls that reach a module, a namespace or an attribute by a computed name.
NAMESPACE_CALLS = ("__import__", "vars", "globals", "locals", "setattr", "delattr")


def harness_tampering(source: str) -> str | None:
  """What in a candidate's source reaches into the harness, or None: a .modules, _getframe or __dict__
  attribute, the name or string base_kernel, HARNESS_TIMER, __main__, __builtins__ or an internals module, a
  __import__, vars, globals, locals, setattr or delattr call, a getattr with a computed name, an import of a
  harness or internals module, or an attribute or subscript assignment whose root name (through attributes,
  subscripts and the function of a call) is an imported module or a flagged name. A write through the
  candidate's own name (out_ref.reshape(1, n)[...] = x) passes. Source that does not parse is left to the
  compile step."""
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
    if isinstance(node, ast.Attribute) and (node.attr in HARNESS_ATTRIBUTES or node.attr in HARNESS_NAMES):
      return f"{ast.unparse(node)} at line {node.lineno}"
    if isinstance(node, ast.Name) and node.id in HARNESS_NAMES:
      return f"{node.id} at line {node.lineno}"
    if isinstance(node, ast.Constant) and node.value in HARNESS_STRINGS:
      return f"the string {node.value!r} at line {node.lineno}"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
      if node.func.id in NAMESPACE_CALLS:
        return f"{node.func.id}() at line {node.lineno}"
      if node.func.id == "getattr" and not (
        len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)
      ):
        return f"getattr() with a computed name at line {node.lineno}"
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
      targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
      for target in targets:
        if not isinstance(target, (ast.Attribute, ast.Subscript)):
          continue
        root = target
        while isinstance(root, (ast.Attribute, ast.Subscript, ast.Call)):
          root = root.func if isinstance(root, ast.Call) else root.value
        # A flagged call at the root (vars(...), __import__(...)) is caught by the call rule above.
        if isinstance(root, ast.Name) and (root.id in imported or root.id in HARNESS_NAMES):
          return f"assignment to {ast.unparse(target)} on {root.id} at line {node.lineno}"
  return None
