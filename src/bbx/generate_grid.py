from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from .process import load_run_encoded
from .viz import load_label_colors, plot_sequence

DEFAULT_SEQUENCE_STEPS: Sequence[int] = (
    0, 1, 2, 5, 10, 25, 50, 100, 200, 500, 750, 1000, 1500, 2000, 4999
)


def parse_sequence_steps(spec: str, total_steps: int,
                         default: Sequence[int] = DEFAULT_SEQUENCE_STEPS) -> list[int]:
    """
    Parse a comma-separated string or 'auto[:N]' instruction into step indices.
    Steps are clamped to [0, total_steps-1] and de-duplicated.
    """
    if total_steps <= 0:
        return []

    spec = spec.strip()
    if not spec:
        candidates: Iterable[int] = default
    elif spec.lower().startswith("auto"):
        parts = spec.split(":")
        count = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 15
        count = max(2, min(count, total_steps))
        if count == 1:
            return [0]
        step = (total_steps - 1) / (count - 1)
        candidates = (int(round(i * step)) for i in range(count))
    else:
        tokens = [t.strip() for t in spec.split(",") if t.strip()]
        candidates = (int(token) for token in tokens)

    steps = sorted({s for s in candidates if 0 <= s < total_steps})
    if not steps:
        return [0] if total_steps == 1 else [0, total_steps - 1]
    if steps[-1] != total_steps - 1:
        steps.append(total_steps - 1)
    return steps


def sequence_panel(states, labels, colors_path: str = "label_colors.json",
                   steps_spec: str = "", save_path: Path | None = None,
                   show: bool = False):
    """
    Plot a panel of selected steps from a run.
    Returns the list of step indices that were rendered.
    """
    steps = parse_sequence_steps(steps_spec, len(states))
    colors = load_label_colors(colors_path)
    plot_sequence(states, labels, colors, steps=steps, save_path=save_path, show=show)
    return steps


def sequence_panel_for_run(run_dir: str | Path,
                           colors_path: str = "label_colors.json",
                           steps_spec: str = "",
                           out_path: Path | None = None,
                           show: bool = False):
    states, labels, _ = load_run_encoded(run_dir)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    steps = sequence_panel(states, labels, colors_path=colors_path,
                           steps_spec=steps_spec, save_path=out_path, show=show)
    return steps, out_path


def main():
    ap = argparse.ArgumentParser(description="Render a montage of CA states from a run directory.")
    ap.add_argument("--run", required=True, help="Run directory, e.g. data/run_000")
    ap.add_argument("--steps", default="", help="Comma list of steps or 'auto[:N]' for evenly spaced.")
    ap.add_argument("--out", default="", help="Optional PNG path to save (default: show only).")
    ap.add_argument("--colors", default="label_colors.json", help="Label→hex mapping JSON.")
    ap.add_argument("--show", action="store_true", help="Display the plot interactively.")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else None
    show = args.show or not out_path
    steps, saved = sequence_panel_for_run(args.run, colors_path=args.colors,
                                          steps_spec=args.steps, out_path=out_path,
                                          show=show)
    print(f"[generate_grid] Rendered steps: {steps}")
    if saved:
        print(f"[generate_grid] Saved montage to {saved}")


if __name__ == "__main__":
    main()
