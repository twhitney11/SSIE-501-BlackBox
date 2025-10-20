from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .process import load_run_encoded
from .viz import load_label_colors, states_to_rgb_frames, save_gif
try:
    from .utils import parse_window
except ImportError:  # pragma: no cover
    from utils import parse_window


def generate_gif(
    run_dir: Path,
    out_path: Path,
    window: str | None,
    stride: int,
    scale: int,
    duration_ms: int,
    loop: int,
    colors_path: Path,
):
    states, labels, _ = load_run_encoded(run_dir)
    total_steps = len(states)
    start, end = parse_window(window, total_steps)
    if start >= end:
        raise ValueError("Empty window for GIF generation")

    selected_states = [states[t] for t in range(start, end, max(1, stride))]
    colors = load_label_colors(colors_path)
    frames = states_to_rgb_frames(selected_states, labels, colors, scale=scale)
    save_gif(frames, out_path, duration_ms=duration_ms, loop=loop)


def main():
    ap = argparse.ArgumentParser(description="Render an animated GIF of a run's steps.")
    ap.add_argument("--run", required=True, help="Run directory (e.g., data/run_000)")
    ap.add_argument("--out", required=True, help="Output GIF path")
    ap.add_argument("--window", default="", help="Time window 'start:end' or 'last:N' (default full run)")
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth frame (default 1)")
    ap.add_argument("--scale", type=int, default=4, help="Pixel up-scale factor (default 4)")
    timing = ap.add_mutually_exclusive_group()
    timing.add_argument("--duration", type=int, default=None, help="Frame duration in ms (default 100)")
    timing.add_argument("--fps", type=float, default=None, help="Frames per second (alternative to --duration)")
    ap.add_argument("--loop", type=int, default=0, help="GIF loop count (0=infinite)")
    ap.add_argument("--colors", default="label_colors.json", help="Path to label→color JSON")
    args = ap.parse_args()

    if args.fps is not None:
        if args.fps <= 0:
            raise ValueError("--fps must be positive")
        duration_ms = max(1, int(round(1000.0 / args.fps)))
    else:
        duration_ms = args.duration if args.duration is not None else 100
        if duration_ms <= 0:
            raise ValueError("--duration must be positive")

    generate_gif(
        run_dir=Path(args.run),
        out_path=Path(args.out),
        window=args.window,
        stride=max(1, args.stride),
        scale=max(1, args.scale),
        duration_ms=duration_ms,
        loop=max(0, args.loop),
        colors_path=Path(args.colors),
    )


if __name__ == "__main__":
    main()
