#!/usr/bin/env python3
"""
collect.py — Fetch CA states from the PHP black box via HTTP and
save them as JSON files under data/run_xxx/.

Exports:
    run_collect(base_url, steps, outdir, run_prefix="run", sleep_ms=0,
                reset=True, seed=None)
"""

import time
import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def fetch_grid(base_url, params=None):
    """Return a 2D list of CSS class labels from <table id="system">."""
    resp = requests.get(base_url, params=params or {})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "system"})
    if not table:
        raise RuntimeError("Could not find <table id='system'> in response HTML")
    grid = []
    for row in table.find_all("tr"):
        cells = []
        for td in row.find_all("td"):
            cls = td.get("class")
            cells.append(cls[0] if cls else "")
        cells and grid.append(cells)
    return grid


def _advance(base_url, n=1, cycles_param="cycles"):
    # advance n cycles (default query param name observed on the page)
    requests.get(base_url, params={cycles_param: int(n)})


def run_collect(
    base_url: str,
    steps: int,
    outdir: Path | str,
    run_prefix: str = "run",
    sleep_ms: int = 0,
    reset: bool = True,
    seed: int | None = None,
):
    """
    Capture steps 0..steps for a single run.
    Creates a new run directory under `outdir` named {run_prefix}_NNN.

    - If `reset` is True, sends `?reset=1` first.
    - If `seed` is provided, includes it in the same reset request (e.g. `?reset=1&seed=123`).
    """
    outroot = Path(outdir)
    outroot.mkdir(parents=True, exist_ok=True)

    # choose next run id
    existing = sorted([p for p in outroot.iterdir() if p.is_dir() and p.name.startswith(run_prefix + "_")])
    next_id = (int(existing[-1].name.split("_")[-1]) + 1) if existing else 0
    run_dir = outroot / f"{run_prefix}_{next_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # optional reset/seed
    if reset:
        params = {"reset": 1}
        if seed is not None:
            params["seed"] = int(seed)
        requests.get(base_url, params=params)

    # capture initial + steps transitions
    for t in range(steps + 1):  # include t=0
        grid = fetch_grid(base_url)
        (run_dir / f"step_{t:04d}.json").write_text(json.dumps(grid))
        if t < steps:
            _advance(base_url, n=1)
            if sleep_ms:
                time.sleep(sleep_ms / 1000.0)

    # optional: small meta
    meta = {"base_url": base_url, "steps": steps, "reset": reset, "seed": seed}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[collect] saved {steps+1} steps → {run_dir}")
    return str(run_dir)
