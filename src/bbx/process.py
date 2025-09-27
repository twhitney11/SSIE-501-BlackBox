#!/usr/bin/env python3
"""
process.py — Load captured black box runs, encode labels to integers,
and provide helpers for neighborhoods, rule learning, and basic analysis.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

# --- Encoding utilities ---

def load_run(run_dir):
    """Load a run directory into a list of 2D lists (raw labels)."""
    run_dir = Path(run_dir)
    steps = sorted(run_dir.glob("step_*.json"))
    states = []
    for step_file in steps:
        with open(step_file) as f:
            states.append(json.load(f))
    return states


def build_label_map(states):
    """Build consistent label -> int mapping from a list of states."""
    labels = sorted({cell for grid in states for row in grid for cell in row})
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    return labels, label_to_idx


def encode_states(raw_states, label_to_idx):
    """Convert list of 2D label grids into numpy arrays of ints."""
    encoded = []
    for grid in raw_states:
        H, W = len(grid), len(grid[0])
        arr = np.zeros((H, W), dtype=int)
        for i in range(H):
            for j in range(W):
                arr[i, j] = label_to_idx[grid[i][j]]
        encoded.append(arr)
    return encoded


# --- Neighborhood utilities ---

def neigh_key(state, i, j, r=1, oob_val=-1):
    """Return flattened neighborhood key (string) for cell (i,j)."""
    H, W = state.shape
    vals = []
    for di in range(-r, r+1):
        for dj in range(-r, r+1):
            ii, jj = i+di, j+dj
            if 0 <= ii < H and 0 <= jj < W:
                vals.append(int(state[ii, jj]))
            else:
                vals.append(oob_val)
    return ",".join(map(str, vals))

def neigh_key_wrap(state, i, j, r=1):
    """Flattened neighborhood with wrap-around (torus)."""
    H, W = state.shape
    vals = []
    for di in range(-r, r+1):
        ii = (i + di) % H
        for dj in range(-r, r+1):
            jj = (j + dj) % W
            vals.append(int(state[ii, jj]))
    return ",".join(map(str, vals))

def apply_rule_wrap(state, rule, r=1):
    """Apply a rulebook assuming wrap-around boundaries."""
    H, W = state.shape
    nxt = np.empty_like(state)
    for i in range(H):
        for j in range(W):
            k = neigh_key_wrap(state, i, j, r=r)
            nxt[i, j] = rule.get(k, state[i, j])
    return nxt

# --- Rule learning ---

def learn_rule(S, T, r=1):
    """Learn mapping from neighborhoods in S to next states in T."""
    H, W = S.shape
    rule = {}
    conflicts = 0
    for i in range(H):
        for j in range(W):
            k = neigh_key(S, i, j, r=r)
            y = int(T[i, j])
            prev = rule.get(k, y)
            rule[k] = prev
            if prev != y:
                conflicts += 1
    return rule, conflicts


def apply_rule(state, rule, r=1):
    """Apply learned rulebook to one state."""
    H, W = state.shape
    nxt = np.empty_like(state)
    for i in range(H):
        for j in range(W):
            k = neigh_key(state, i, j, r=r)
            nxt[i, j] = rule.get(k, state[i, j])  # fallback: copy itself
    return nxt


# --- Basic analysis helpers ---

def diff_count(A, B):
    """Number of mismatched cells between two states."""
    return int((A != B).sum())


def coverage(states, r=1):
    """Count unique neighborhoods across a list of states."""
    seen = Counter()
    for S in states:
        H, W = S.shape
        for i in range(H):
            for j in range(W):
                k = neigh_key(S, i, j, r=r)
                seen[k] += 1
    return len(seen), seen

def summarize_run(run_dir, radius=1):
    """Load a run, encode it, and print a quick summary."""
    raw_states = load_run(run_dir)
    labels, label_to_idx = build_label_map(raw_states)
    enc_states = encode_states(raw_states, label_to_idx)

    print(f"=== Summary for {run_dir} ===")
    print(f"Steps: {len(enc_states)} | Grid size: {enc_states[0].shape}")
    print(f"Labels: {len(labels)} -> {labels}")

    # Unique neighborhoods seen
    uniq_count, seen = coverage(enc_states, r=radius)
    print(f"Unique neighborhoods (r={radius}): {uniq_count}")

    # Conflicts per step
    total_conf, total_mismatch = 0, 0
    for t in range(len(enc_states)-1):
        S, T = enc_states[t], enc_states[t+1]
        rule, conf = learn_rule(S, T, r=radius)
        pred = apply_rule(S, rule, r=radius)
        mismatches = diff_count(pred, T)
        if conf or mismatches:
            print(f"Step {t}->{t+1}: conflicts={conf}, mismatches={mismatches}")
        total_conf += conf
        total_mismatch += mismatches

    print(f"Total conflicts: {total_conf}, Total mismatches: {total_mismatch}")
    print()

def summarize_runs(run_dirs, radius=1):
    """Aggregate stats across multiple runs."""
    all_states = []
    label_sets = set()

    for run_dir in run_dirs:
        raw_states = load_run(run_dir)
        labels, label_to_idx = build_label_map(raw_states)
        label_sets.update(labels)
        enc_states = encode_states(raw_states, label_to_idx)
        all_states.extend(enc_states)

    print(f"=== Aggregate summary for {len(run_dirs)} runs ===")
    print(f"Labels total: {len(label_sets)}")

    uniq_count, seen = coverage(all_states, r=radius)
    print(f"Unique neighborhoods (r={radius}): {uniq_count}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python process.py <run_dir> [<run_dir2> ...]")
    else:
        run_dirs = sys.argv[1:]
        if len(run_dirs) == 1:
            summarize_run(run_dirs[0], radius=1)
        else:
            # Do per-run summaries
            for rd in run_dirs:
                summarize_run(rd, radius=1)
            # Then do aggregate
            summarize_runs(run_dirs, radius=1)

# --- process.py additions ---

def cell_type(i, j, H, W):
    if (i in (0, H-1)) and (j in (0, W-1)): return "corner"
    if i in (0, H-1) or j in (0, W-1):      return "edge"
    return "interior"

def learn_rule_scoped(S, T, r=1, scope=None):
    """Learn rule with an optional scope filter ((i,j)->bool)."""
    H, W = S.shape
    rule, conflicts = {}, 0
    for i in range(H):
        for j in range(W):
            if scope and not scope(i, j): 
                continue
            k = neigh_key(S, i, j, r=r)
            y = int(T[i, j])
            prev = rule.get(k, y)
            rule[k] = prev
            if prev != y: conflicts += 1
    return rule, conflicts

def count_conflicts_over_series(states, r=1, scope=None):
    """Merge transitions across all t and count conflicts using one rulebook."""
    rule, conflicts = {}, 0
    for t in range(len(states)-1):
        S, T = states[t], states[t+1]
        H, W = S.shape
        for i in range(H):
            for j in range(W):
                if scope and not scope(i, j):
                    continue
                k = neigh_key(S, i, j, r=r)
                y = int(T[i, j])
                if k in rule and rule[k] != y: conflicts += 1
                else: rule[k] = y
    return rule, conflicts

def load_run_encoded(run_dir):
    """
    Load one run directory (data/run_000, etc).
    Returns: (states, labels, label_to_idx)
      - states: list of H×W int arrays (encoded by label index)
      - labels: list of class names (sorted consistently)
      - label_to_idx: dict {label: index}
    """
    run_dir = Path(run_dir)

    # discover all labels across all step files
    all_labels = set()
    stepfiles = sorted(run_dir.glob("step_*.json"))
    if not stepfiles:
        raise FileNotFoundError(f"No step_*.json files in {run_dir}")

    for stepfile in stepfiles:
        grid = json.loads(stepfile.read_text())
        for row in grid:
            all_labels.update(row)

    labels = sorted(all_labels)  # consistent order
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    # encode states
    states = []
    for stepfile in stepfiles:
        grid = json.loads(stepfile.read_text())
        H, W = len(grid), len(grid[0])
        S = np.empty((H, W), dtype=np.int32)
        for i in range(H):
            for j in range(W):
                lab = grid[i][j]
                S[i, j] = label_to_idx[lab]
        states.append(S)

    return states, labels, label_to_idx


# --- Example pipeline and testing ---
'''if __name__ == "__main__":
    run_path = "data/run_000"
    raw_states = load_run(run_path)
    labels, label_to_idx = build_label_map(raw_states)
    enc_states = encode_states(raw_states, label_to_idx)

    print(f"Loaded {len(enc_states)} steps from {run_path}")
    print(f"Label set: {labels}")

    # Learn a rule from step 0 -> 1
    rule, conf = learn_rule(enc_states[0], enc_states[1], r=1)
    print(f"Learned rule with {len(rule)} entries, {conf} conflicts")

    # Test replay
    pred1 = apply_rule(enc_states[0], rule, r=1)
    mismatches = diff_count(pred1, enc_states[1])
    print(f"Mismatches when replaying step 0->1: {mismatches}")
'''
