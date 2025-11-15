from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from .process import load_run, load_run_encoded
try:
    from .utils import parse_window
except ImportError:  # pragma: no cover
    from utils import parse_window  # type: ignore


def export_runs_to_csv(run_dirs: Sequence[str], out_path: str | Path,
                       window: str = "", include_encoded: bool = False) -> Path:
    if not run_dirs:
        raise ValueError("At least one run directory must be provided.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["run", "step", "i", "j", "label"]
    if include_encoded:
        header.append("label_idx")

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for run_dir in run_dirs:
            run_dir = str(run_dir)
            run_name = Path(run_dir).name
            if include_encoded:
                _export_encoded_run(run_dir, run_name, writer, window)
            else:
                _export_raw_run(run_dir, run_name, writer, window)

    return out_path


def _export_raw_run(run_dir: str, run_name: str, writer: csv.writer, window: str) -> None:
    states = load_run(run_dir)
    if not states:
        return
    start, end = parse_window(window, len(states)) if window else (0, len(states))
    for step_idx in range(start, end):
        grid = states[step_idx]
        for i, row in enumerate(grid):
            for j, label in enumerate(row):
                writer.writerow([run_name, step_idx, i, j, label])


def _export_encoded_run(run_dir: str, run_name: str, writer: csv.writer, window: str) -> None:
    states, labels, _ = load_run_encoded(run_dir)
    if not states:
        return
    start, end = parse_window(window, len(states)) if window else (0, len(states))
    for step_idx in range(start, end):
        grid = states[step_idx]
        H, W = grid.shape
        for i in range(H):
            for j in range(W):
                idx = int(grid[i, j])
                label = labels[idx]
                writer.writerow([run_name, step_idx, i, j, label, idx])
