from __future__ import annotations

from typing import Tuple


def parse_window(window: str | None, total: int) -> Tuple[int, int]:
    """
    Parse window spec into (start, end) indices.
    Supports:
      - None / ""  -> (0, total)
      - "start:end"
      - "last:N"
      - "start"    -> from start to total
    """
    if not window:
        return 0, total

    window = window.strip()
    if window.startswith("last:"):
        try:
            n = int(window.split(":", 1)[1])
        except ValueError:
            raise ValueError(f"Invalid window specification: {window}")
        return max(0, total - n), total

    if ":" in window:
        start_str, end_str = window.split(":", 1)
        try:
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else total
        except ValueError:
            raise ValueError(f"Invalid window specification: {window}")
    else:
        try:
            start = int(window)
        except ValueError:
            raise ValueError(f"Invalid window specification: {window}")
        end = total

    start = max(0, min(start, total))
    end = max(start + 1, min(end, total)) if start < total else total
    return start, end
