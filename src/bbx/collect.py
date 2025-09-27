#!/usr/bin/env python3
"""
collect.py — Fetch CA states from the PHP black box via HTTP,
save them as JSON for later analysis.
"""

import argparse
import os
import time
import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path


def fetch_grid(base_url, params=None):
    """Fetch the HTML table from BlackBox.php and parse it into a 2D list of labels."""
    resp = requests.get(base_url, params=params)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "system"})
    if not table:
        raise RuntimeError("Could not find <table id='system'> in response")

    grid = []
    for row in table.find_all("tr"):
        cells = [td.get("class")[0] if td.get("class") else "" for td in row.find_all("td")]
        grid.append(cells)
    return grid


def run_capture(base_url, outdir, steps=100, reset=True, delay=0.0):
    """Run one capture of N steps, saving each grid as JSON."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Reset system if requested
    if reset:
        print("Resetting system...")
        requests.get(base_url, params={"reset": 1})

    for t in range(steps+1):  # +1 to capture initial state
        grid = fetch_grid(base_url)
        step_file = outdir / f"step_{t:04d}.json"
        with open(step_file, "w") as f:
            json.dump(grid, f)
        print(f"Saved step {t} -> {step_file}")
        if t < steps:
            # Advance one step
            requests.get(base_url, params={"cycles": 1})
            if delay:
                time.sleep(delay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Base URL to BlackBox.php (e.g. http://localhost/BlackBox.php)")
    ap.add_argument("--runs", type=int, default=1, help="Number of runs to perform")
    ap.add_argument("--steps", type=int, default=100, help="Number of steps per run")
    ap.add_argument("--out", default="data", help="Output directory for captured data")
    ap.add_argument("--delay", type=float, default=0.0, help="Delay (s) between steps to avoid server overload")
    args = ap.parse_args()

    base_url = args.url
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    for r in range(args.runs):
        run_dir = root / f"run_{r:03d}"
        print(f"=== Starting run {r}, saving to {run_dir} ===")
        run_capture(base_url, run_dir, steps=args.steps, reset=True, delay=args.delay)


if __name__ == "__main__":
    main()
