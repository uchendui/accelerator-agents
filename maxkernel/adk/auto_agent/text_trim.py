"""Helpers for keeping model prompt text bounded."""


def trim_to_tail(text: str, max_chars: int) -> str:
  if len(text) <= max_chars:
    return text
  dropped_chars = len(text) - max_chars
  return f"[... {dropped_chars} characters dropped ...]\n{text[-max_chars:]}"
