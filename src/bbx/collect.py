# collect.py
import argparse
import time
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import numpy as np
import hashlib

STEP_RE = re.compile(r"Current step:\s*(\d+)", re.I)

def _parse_grid_and_step(html: str):
    soup = BeautifulSoup(html, "html.parser")
    # grid
    table = soup.find("table", {"id": "system"})
    if not table:
        raise RuntimeError("Could not find <table id='system'> in response")
    grid = []
    for row in table.find_all("tr"):
        cells = [td.get("class")[0] if td.get("class") else "" for td in row.find_all("td")]
        grid.append(cells)
    # step
    ctrl = soup.find(id="controls")
    step = None
    if ctrl:
        m = STEP_RE.search(ctrl.get_text(" ", strip=True))
        if m:
            step = int(m.group(1))
    return grid, step

def _hash_grid(grid):
    # stable hash of labels to detect accidental reseed/no-advance
    arr = np.array(grid)
    h = hashlib.sha256(arr.tobytes()).hexdigest()
    return h

def _get(session: requests.Session, base_url: str, params=None):
    headers = {"Referer": base_url}
    resp = session.get(base_url, params=params or {}, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.text

def run_collect(base_url: str, steps: int, outdir: Path, run_prefix: str = "run",
                sleep_ms: int = 0, reset: bool = True, seed: int | None = None):
    """
    Capture a single run of N steps from the PHP black box.
    - Keeps one HTTP session so the server retains state.
    - Optionally seeds and resets once at the start.
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    # choose a free run dir
    n = 0
    while True:
        run_dir = outdir / f"{run_prefix}_{n:03d}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True)
            break
        n += 1
    print(f"=== Starting capture → {run_dir} ===")

    s = requests.Session()

    # Optional: reset & seed once
    if reset:
        params = {"reset": 1}
        if seed is not None:
            params["seed"] = int(seed)
        html = _get(s, base_url, params=params)
    else:
        html = _get(s, base_url)

    # Capture initial state
    grid, step_no = _parse_grid_and_step(html)
    if step_no is None:
        print("[warn] Could not read 'Current step' on first page; continuing anyway")
        step_no = 0

    prev_hash = _hash_grid(grid)
    (run_dir / f"step_{0:04d}.json").write_text(json.dumps(grid))
    print(f"Saved step 0 (server step={step_no})")

    # Advance loop
    delay = max(0.0, float(sleep_ms) / 1000.0)
    last_server_step = step_no
    for t in range(1, steps + 1):
        # advance by one cycle; some deployments use 'cycles', some expect both
        html = _get(s, base_url, params={"cycles": 1})
        grid, step_no = _parse_grid_and_step(html)

        # sanity checks
        if step_no is not None and step_no < last_server_step:
            print(f"[warn] server step decreased ({last_server_step} → {step_no}) — possible reseed")
        if step_no is not None and step_no == last_server_step:
            print(f"[warn] server step did not advance ({step_no})")
        last_server_step = step_no if step_no is not None else last_server_step

        h = _hash_grid(grid)
        if h == prev_hash:
            print(f"[warn] grid hash unchanged at t={t} (possible no-advance)")
        prev_hash = h

        # save
        (run_dir / f"step_{t:04d}.json").write_text(json.dumps(grid))
        if step_no is not None:
            print(f"Saved step {t} (server step={step_no})")
        else:
            print(f"Saved step {t}")

        if t < steps and delay:
            time.sleep(delay)

    # meta for traceability
    meta = {
        "base_url": base_url,
        "steps": steps,
        "seed": seed,
        "reset": reset,
        "sleep_ms": sleep_ms,
        "started_run_index": n,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {run_dir}")
