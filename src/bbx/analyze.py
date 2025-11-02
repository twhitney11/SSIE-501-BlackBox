#!/usr/bin/env python3

import argparse
import json
import random
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
import hashlib

from .process import neigh_key_wrap, apply_rule_wrap
from .utils import parse_window
from .best_k import summarise_period_scan

# Set bg color to gray because "mex" is white and invisible.
plt.rcParams['axes.facecolor']   = '#cccccc' 
plt.rcParams['figure.facecolor'] = '#cccccc'
plt.rcParams['savefig.facecolor'] = '#cccccc'


# ---- import local helpers from process.py ----
from .process import (
    load_run, build_label_map, encode_states,
    learn_rule, apply_rule, diff_count, neigh_key,
    learn_rule_scoped, count_conflicts_over_series
)

# Setting some constants for black and white classes
BLACK, WHITE = "gru", "mex"

# --------------- IO utils ---------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))

def save_csv(rows, path: Path):
    # rows: list[list|tuple]
    lines = []
    for r in rows:
        line = []
        for x in r:
            s = "" if x is None else str(x)
            if any(c in s for c in [",", '"', "\n"]):
                s = '"' + s.replace('"', '""') + '"'
            line.append(s)
        lines.append(",".join(line))
    path.write_text("\n".join(lines))


def _to_bool(val) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return bool(val)
    if isinstance(val, (float, np.floating)):
        return bool(int(val))
    s = str(val).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n", ""}:
        return False
    return True


def load_scope_mask_csv(path: Path, column: str | None = None) -> np.ndarray:
    df = pd.read_csv(path)
    if "i" not in df.columns or "j" not in df.columns:
        raise ValueError(f"Scope mask file {path} must contain 'i' and 'j' columns.")
    if column is None:
        candidates = [c for c in df.columns if c not in {"i", "j"}]
        if not candidates:
            raise ValueError(f"Scope mask file {path} needs a boolean column besides 'i','j'.")
        column = candidates[0]
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {path}.")
    rows = df[["i", "j", column]].to_numpy()
    H = int(df["i"].max()) + 1
    W = int(df["j"].max()) + 1
    mask = np.zeros((H, W), dtype=bool)
    for i, j, v in rows:
        mask[int(i), int(j)] = _to_bool(v)
    return mask


def load_scope_region_mask(path: Path, region: str) -> np.ndarray:
    df = pd.read_csv(path)
    if {"i", "j", "region"} - set(df.columns):
        raise ValueError(f"Region labels file {path} must include 'i','j','region'.")
    rows = df[["i", "j", "region"]].to_numpy()
    H = int(df["i"].max()) + 1
    W = int(df["j"].max()) + 1
    mask = np.zeros((H, W), dtype=bool)
    region = region.strip()
    for i, j, reg in rows:
        mask[int(i), int(j)] = str(reg).strip() == region
    return mask

# --------------- loading & encoding ---------------

def load_runs_encoded(run_dirs, window: str | None = None):
    """Return (all_states_encoded, labels_global, label_to_idx_global, per_run_states)"""
    per_run_raw = []
    for rd in run_dirs:
        raw = load_run(rd)
        if not raw:
            print(f"[warn] empty run: {rd}")
            continue
        start, end = parse_window(window, len(raw)) if raw else (0, 0)
        sliced = raw[start:end]
        if not sliced:
            print(f"[warn] window {window} produced no steps for {rd}; skipping")
            continue
        per_run_raw.append(sliced)

    if not per_run_raw:
        raise ValueError("No runs contained data for the requested window")

    # Build a global label map across all runs (stable, alphabetical)
    all_raw = [grid for run in per_run_raw for grid in run]
    labels, label_to_idx = build_label_map(all_raw)

    # Encode each run using the global map
    per_run_enc = []
    for raw in per_run_raw:
        enc = encode_states(raw, label_to_idx)
        per_run_enc.append(enc)

    all_states = [arr for run in per_run_enc for arr in run]
    return all_states, labels, label_to_idx, per_run_enc

def period_scan(per_run_states, r=1, max_k=32, split="none", scope_mask: np.ndarray | None = None):
    """
    Scan time period k (2..max_k) and optionally split by region.
    split ∈ {"none","edge","corner_edge"}.
    Returns CSV-like rows: [k, split, phase, region, conflicts, rule_size]
    """
    rows = [["k","split","phase","region","conflicts","rule_size"]]
    if not per_run_states:
        return rows

    states = per_run_states[0]
    H, W = states[0].shape

    def region_scope(region):
        if split == "none":
            return lambda i, j: True
        if split == "edge":
            if region == "interior":
                return lambda i, j: (i not in (0, H-1) and j not in (0, W-1))
            if region == "edge":
                return lambda i, j: (i in (0, H-1) or j in (0, W-1))
        if split == "corner_edge":
            if region == "corner":
                return lambda i, j: (i in (0, H-1) and j in (0, W-1))
            if region == "edge":
                return lambda i, j: ((i in (0, H-1) or j in (0, W-1)) and not (i in (0, H-1) and j in (0, W-1)))
            if region == "interior":
                return lambda i, j: (i not in (0, H-1) and j not in (0, W-1))
        return lambda i, j: True

    regions_by_split = {
        "none": ["all"],
        "edge": ["interior", "edge"],
        "corner_edge": ["interior", "edge", "corner"],
    }

    for k in range(2, max_k + 1):
        for region in regions_by_split.get(split, ["all"]):
            scope = None if region == "all" else region_scope(region)
            for phase in range(k):
                rule = {}
                conflicts = 0
                for t in range(len(states) - 1):
                    if t % k != phase:
                        continue
                    S, T = states[t], states[t + 1]
                    for i in range(H):
                        for j in range(W):
                            if scope_mask is not None and not scope_mask[i, j]:
                                continue
                            if scope and not scope(i, j):
                                continue
                            key = neigh_key(S, i, j, r=r)
                            y = int(T[i, j])
                            if key in rule and rule[key] != y:
                                conflicts += 1
                            else:
                                rule[key] = y
                rows.append([k, split, phase, region, conflicts, len(rule)])
    return rows


# --------------- coverage & conflicts ---------------

def neighborhood_counts(states, r=1, oob_val=-1, scope_mask: np.ndarray | None = None):
    seen = Counter()
    for S in states:
        H, W = S.shape
        for i in range(H):
            for j in range(W):
                if scope_mask is not None and not scope_mask[i, j]:
                    continue
                vals = []
                for di in range(-r, r+1):
                    for dj in range(-r, r+1):
                        ii, jj = i+di, j+dj
                        if 0 <= ii < H and 0 <= jj < W:
                            vals.append(int(S[ii, jj]))
                        else:
                            vals.append(oob_val)
                seen[",".join(map(str, vals))] += 1
    return seen

def aggregate_conflicts(states, r=1, scope_mask: np.ndarray | None = None):
    """Merge transitions across all consecutive pairs; count conflicts and build a global rulebook."""
    scope = (lambda i, j: scope_mask[i, j]) if scope_mask is not None else None
    return count_conflicts_over_series(states, r=r, scope=scope)


def masked_diff_count(A: np.ndarray, B: np.ndarray, mask: np.ndarray | None = None) -> int:
    if mask is None:
        return diff_count(A, B)
    return int((A != B)[mask].sum())

def extract_conflict_examples(states, labels, r=1, max_examples=50, scope_mask: np.ndarray | None = None):
    """Return list of dicts with neighborhood (as label grid) and outputs observed."""
    H, W = states[0].shape
    seen = {}
    examples = []
    for t in range(len(states) - 1):
        S, T = states[t], states[t + 1]
        for i in range(H):
            for j in range(W):
                if scope_mask is not None and not scope_mask[i, j]:
                    continue
                # build key with OOB so edges are distinct
                vals = []
                for di in range(-r, r+1):
                    for dj in range(-r, r+1):
                        ii, jj = i+di, j+dj
                        if 0 <= ii < H and 0 <= jj < W:
                            vals.append(int(S[ii, jj]))
                        else:
                            vals.append(-1)
                key = ",".join(map(str, vals))
                y = int(T[i, j])
                if key not in seen:
                    seen[key] = {y}
                else:
                    if y not in seen[key]:
                        seen[key].add(y)
                        # conflict discovered
                        # decode neighborhood for export
                        lab_map = {i: lab for i, lab in enumerate(labels)}
                        lab_map[-1] = "OOB"
                        size = 2 * r + 1
                        vals_int = list(map(int, key.split(",")))
                        grid = []
                        s = 0
                        for _ in range(size):
                            row = []
                            for _ in range(size):
                                row.append(lab_map[vals_int[s]])
                                s += 1
                            grid.append(row)
                        examples.append({
                            "t": t,
                            "i": i,
                            "j": j,
                            "radius": r,
                            "neighborhood_labels": grid,
                            "outputs_labels": [labels[o] for o in seen[key] if o != -1]
                        })
                        if len(examples) >= max_examples:
                            return examples
    return examples

# --------------- permutation invariance ---------------

def permutation_invariance_test(states, labels, trials=20, r=1, seed=42, scope_mask: np.ndarray | None = None):
    """Shuffle label ids and see if step-accuracy for 0->1 changes; repeat across early steps."""
    rng = random.Random(seed)
    L = len(labels)
    results = []
    steps_to_test = list(range(0, min(10, len(states) - 1)))  # first 10 transitions
    scope = (lambda i, j: scope_mask[i, j]) if scope_mask is not None else None
    for t in steps_to_test:
        S, T = states[t], states[t+1]
        if scope:
            base_rule, _ = learn_rule_scoped(S, T, r=r, scope=scope)
        else:
            base_rule, _ = learn_rule(S, T, r=r)
        base_pred = apply_rule(S, base_rule, r=r)
        base_mis = masked_diff_count(base_pred, T, scope_mask)
        worse = same = better = 0
        for _ in range(trials):
            perm = list(range(L)); rng.shuffle(perm)
            Sperm = np.vectorize(lambda x: perm[x])(S)
            Tperm = np.vectorize(lambda x: perm[x])(T)
            if scope:
                rb, _ = learn_rule_scoped(Sperm, Tperm, r=r, scope=scope)
            else:
                rb, _ = learn_rule(Sperm, Tperm, r=r)
            pred = apply_rule(Sperm, rb, r=r)
            mis = masked_diff_count(pred, Tperm, scope_mask)
            if mis > base_mis: worse += 1
            elif mis < base_mis: better += 1
            else: same += 1
        results.append({
            "step": f"{t}->{t+1}",
            "baseline_mismatches": int(base_mis),
            "trials": trials,
            "worse": worse, "same": same, "better": better
        })
    return results

# --------------- totalistic vs positional ---------------

def totalistic_key(vals):
    cnt = Counter(vals)
    # stable by label id; include OOB (-1)
    return "|".join(f"{k}:{cnt[k]}" for k in sorted(cnt.keys()))

def learn_totalistic(S, T, r=1, oob=-1, scope_mask: np.ndarray | None = None):
    H, W = S.shape
    rule = {}
    conflicts = 0
    for i in range(H):
        for j in range(W):
            if scope_mask is not None and not scope_mask[i, j]:
                continue
            vals = []
            for di in range(-r, r+1):
                for dj in range(-r, r+1):
                    ii, jj = i+di, j+dj
                    vals.append(int(S[ii, jj]) if 0 <= ii < H and 0 <= jj < W else oob)
            k = totalistic_key(vals)
            y = int(T[i, j])
            if k in rule and rule[k] != y:
                conflicts += 1
            else:
                rule[k] = y
    return rule, conflicts

def apply_totalistic(S, rule, r=1, oob=-1):
    H, W = S.shape
    T = np.empty_like(S)
    for i in range(H):
        for j in range(W):
            vals = []
            for di in range(-r, r+1):
                for dj in range(-r, r+1):
                    ii, jj = i+di, j+dj
                    vals.append(int(S[ii, jj]) if 0 <= ii < H and 0 <= jj < W else oob)
            k = totalistic_key(vals)
            T[i, j] = rule.get(k, S[i, j])
    return T

def compare_totalistic_vs_positional(states, r=1, scope_mask: np.ndarray | None = None):
    """Return table of mismatches over first K transitions for both models."""
    K = min(20, len(states) - 1)
    rows = [["step", "total_cells", "mismatch_positional", "mismatch_totalistic"]]
    scope = (lambda i, j: scope_mask[i, j]) if scope_mask is not None else None
    for t in range(K):
        S, T = states[t], states[t+1]
        if scope:
            rb_pos, _ = learn_rule_scoped(S, T, r=r, scope=scope)
        else:
            rb_pos, _ = learn_rule(S, T, r=r)
        pred_pos = apply_rule(S, rb_pos, r=r)
        mis_pos = masked_diff_count(pred_pos, T, scope_mask)

        rb_tot, _ = learn_totalistic(S, T, r=r, scope_mask=scope_mask)
        pred_tot = apply_totalistic(S, rb_tot, r=r)
        mis_tot = masked_diff_count(pred_tot, T, scope_mask)

        rows.append([f"{t}->{t+1}", S.size, mis_pos, mis_tot])
    return rows

# --------------- plotting ---------------

def plot_mismatch_curve(states, r=1, outdir: Path = None, prefix="", scope_mask: np.ndarray | None = None):
    """Train on step t->t+1 and report exact replay mismatch for each of first K transitions."""
    K = min(100, len(states) - 1)
    xs = []
    ms = []
    scope = (lambda i, j: scope_mask[i, j]) if scope_mask is not None else None
    for t in range(K):
        S, T = states[t], states[t+1]
        if scope:
            rb, _ = learn_rule_scoped(S, T, r=r, scope=scope)
        else:
            rb, _ = learn_rule(S, T, r=r)
        pred = apply_rule(S, rb, r=r)
        ms.append(masked_diff_count(pred, T, scope_mask))
        xs.append(t)
    plt.figure()
    plt.plot(xs, ms, marker='.')
    plt.xlabel("step t (replay of t→t+1)")
    plt.ylabel("# mismatches")
    plt.title(f"Exact replay mismatches (r={r})")
    plt.grid(True, alpha=0.3)
    if outdir:
        p = outdir / f"{prefix}mismatch_curve_r{r}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"[saved] {p}")
    plt.close()

def plot_label_histogram(states, labels, colors, outdir: Path = None, prefix="",
                         scope_mask: np.ndarray | None = None, filename_suffix: str = ""):
    title="Label histogram"
    import matplotlib.pyplot as plt
    import numpy as np

    S0 = states[0]
    S1 = states[-1]
    if scope_mask is not None:
        if scope_mask.shape != S0.shape:
            raise ValueError("Scope mask shape mismatch for label histogram")
        mask = scope_mask.astype(bool)
        if mask.sum() == 0:
            print("[warn] scope mask for histogram selects zero cells; skipping plot")
            return
        init_counts = np.bincount(S0[mask].ravel(), minlength=len(labels))
        final_counts = np.bincount(S1[mask].ravel(), minlength=len(labels))
    else:
        init_counts = np.bincount(S0.ravel(), minlength=len(labels))
        final_counts = np.bincount(S1.ravel(), minlength=len(labels))

    # Sort labels by final counts
    order = np.argsort(-final_counts)  # descending
    labels_sorted = [labels[i] for i in order]
    init_sorted = init_counts[order]
    final_sorted = final_counts[order]

    x = np.arange(len(labels_sorted))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10,5))
    bars1 = ax.bar(x - width/2, init_sorted, width, label="Initial",
                   color=[colors.get(lab, "#808080") for lab in labels_sorted])
    bars2 = ax.bar(x + width/2, final_sorted, width, label="Final",
                   color=[colors.get(lab, "#808080") for lab in labels_sorted])

    # Add text labels on top of bars
    for bars in [bars1, bars2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{int(b.get_height())}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted)
    ax.set_ylabel("Count")
    ax.set_title(prefix + title)
    ax.legend()

    fig.tight_layout()
    if outdir:
        suffix = filename_suffix or ""
        outpath = Path(outdir) / f"{prefix}label_histogram{suffix}.png"
        fig.savefig(outpath, dpi=150, facecolor="#f0f0f0")
        print(f"[saved] {outpath}")
    else:
        plt.show()


# ----- SHAPE DIAGNOSTICS (ring/border evidence) -----

def chebyshev_distance_from_edge(H, W):
    I, J = np.indices((H, W))
    d_top, d_left = I, J
    d_bottom, d_right = (H-1)-I, (W-1)-J
    return np.minimum(np.minimum(d_top, d_bottom), np.minimum(d_left, d_right))

def black_radial_profile(states, labels, black_label=BLACK):
    """Fraction of black at each Chebyshev distance d from the edge, per step."""
    black_idx = labels.index(black_label)
    H, W = states[0].shape
    D = chebyshev_distance_from_edge(H, W)
    maxd = int(D.max())
    rows = []
    for t, S in enumerate(states):
        for d in range(maxd + 1):
            mask = (D == d)
            frac = (S[mask] == black_idx).mean()
            rows.append({"step": t, "d": d, "black_fraction": float(frac)})
    return pd.DataFrame(rows)

def plot_black_radial_profile(df_rad, out_png: Path, title="Black fraction vs distance from edge"):
    """Heat-map like lineplot: average across steps."""
    piv = df_rad.groupby("d")["black_fraction"].mean()
    plt.figure(figsize=(8,4))
    plt.plot(piv.index.values, piv.values, marker='o')
    plt.xlabel("Chebyshev distance from edge (cells)")
    plt.ylabel("Mean black fraction")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_png}")

def black_by_row_col(states, labels, black_label=BLACK):
    """Row/column peak locations & magnitudes for black, per step."""
    black_idx = labels.index(black_label)
    rows = []
    for t, S in enumerate(states):
        by_row = (S == black_idx).mean(axis=1)
        by_col = (S == black_idx).mean(axis=0)
        rows.append({
            "step": t,
            "row_max": float(by_row.max()),
            "row_argmax": int(by_row.argmax()),
            "col_max": float(by_col.max()),
            "col_argmax": int(by_col.argmax())
        })
    return pd.DataFrame(rows)

def plot_row_col_traces(df_rc, H, W, out_png: Path, title="Black row/col peaks over time"):
    plt.figure(figsize=(10,4))
    ax1 = plt.subplot(1,2,1)
    ax1.plot(df_rc["step"], df_rc["row_argmax"], marker='.', linewidth=0.7)
    ax1.set_title("Row of max black fraction"); ax1.set_xlabel("Step"); ax1.set_ylabel("Row idx")
    ax1.set_ylim(-1, H)
    ax2 = plt.subplot(1,2,2)
    ax2.plot(df_rc["step"], df_rc["col_argmax"], marker='.', linewidth=0.7)
    ax2.set_title("Col of max black fraction"); ax2.set_xlabel("Step"); ax2.set_ylabel("Col idx")
    ax2.set_ylim(-1, W)
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_png}")

def band_mask(H, W, d0, d1):
    """Boolean mask for a ring/band between distances [d0, d1] from the edge."""
    D = chebyshev_distance_from_edge(H, W)
    return (D >= d0) & (D <= d1)

def band_vs_interior_fractions(states, labels, d0=2, d1=3, black_label=BLACK, white_label=WHITE):
    """Compare black/white fractions in a band vs the interior."""
    idx_black = labels.index(black_label)
    idx_white = labels.index(white_label)
    H, W = states[0].shape
    band = band_mask(H, W, d0, d1)
    interior = ~band
    rows = []
    for t, S in enumerate(states):
        b_frac_black = (S[band] == idx_black).mean()
        b_frac_white = (S[band] == idx_white).mean()
        i_frac_black = (S[interior] == idx_black).mean()
        i_frac_white = (S[interior] == idx_white).mean()
        rows.append({
            "step": t,
            "band_black": float(b_frac_black),
            "band_white": float(b_frac_white),
            "interior_black": float(i_frac_black),
            "interior_white": float(i_frac_white),
        })
    return pd.DataFrame(rows)

def plot_band_vs_interior(df_band, out_png: Path, d0=2, d1=3, title=None):
    title = title or f"Band (d∈[{d0},{d1}]) vs interior — black/white fractions"
    plt.figure(figsize=(10,4))
    # smol smoothing for readability
    def smooth(y, w=9):
        y = pd.Series(y)
        return y.rolling(w, center=True, min_periods=1).mean().values
    steps = df_band["step"].values
    plt.plot(steps, smooth(df_band["band_black"]), label=f"band black", linewidth=1.5)
    plt.plot(steps, smooth(df_band["interior_black"]), label=f"interior black", linewidth=1.0, linestyle="--")
    plt.plot(steps, smooth(df_band["band_white"]), label=f"band white", linewidth=1.5)
    plt.plot(steps, smooth(df_band["interior_white"]), label=f"interior white", linewidth=1.0, linestyle="--")
    plt.xlabel("Step"); plt.ylabel("Fraction"); plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_png}")

def fractions_time_series(states, labels):
    """Return dict label->np.array of fractions over time."""
    import numpy as np
    H, W = states[0].shape
    total = H*W
    L = len(labels)
    out = {lab: np.zeros(len(states), dtype=float) for lab in labels}
    for t, S in enumerate(states):
        vals, counts = np.unique(S, return_counts=True)
        freq = dict(zip(vals, counts))
        for idx, lab in enumerate(labels):
            out[lab][t] = freq.get(idx, 0) / total
    return out

def l_inf_front_radius(S, black_idx):
    H, W = S.shape
    rows = np.where((S == black_idx).any(axis=1))[0]
    cols = np.where((S == black_idx).any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0: return 0
    d_top = rows.min()
    d_bot = (H-1) - rows.max()
    d_left = cols.min()
    d_right = (W-1) - cols.max()
    return int(min(d_top, d_bot, d_left, d_right))

def ring_radius_series(states, labels):
    import numpy as np
    bidx = labels.index(BLACK)
    return np.array([l_inf_front_radius(S, bidx) for S in states], dtype=int)

def autocorr_1d(x, max_lag=64):
    import numpy as np
    x = np.asarray(x, dtype=float)
    x = (x - x.mean())
    denom = (x**2).sum()
    if denom == 0: return np.zeros(max_lag+1)
    ac = np.correlate(x, x, mode="full")
    mid = len(ac)//2
    ac = ac[mid:mid+max_lag+1] / denom
    return ac  # ac[0] = 1.0

def collapse_bw_other(S, labels):
    """Map labels→{0:black,1:white,2:other} on an int grid."""
    import numpy as np
    b = labels.index(BLACK); w = labels.index(WHITE)
    out = np.full_like(S, 2, dtype=np.uint8)
    out[S == b] = 0
    out[S == w] = 1
    return out

def hamming_fraction(A, B):
    import numpy as np
    return float((A != B).sum()) / A.size

def parse_perturb_spec(spec_str):
    """
    'block:cx=10,cy=10,w=5,h=5,mode=rand,steps=100,base_step=-100'
    modes: rand, flipbw, setwhite, setblack
    """
    out = {}
    if not spec_str: return None
    kind, *rest = spec_str.split(":", 1)
    out["kind"] = kind
    if rest:
        for kv in rest[0].split(","):
            if not kv: continue
            k, v = kv.split("=")
            if k in {"cx","cy","w","h","steps","base_step"}:
                out[k] = int(v)
            else:
                out[k] = v
    # defaults
    out.setdefault("mode", "rand")
    out.setdefault("steps", 100)
    out.setdefault("base_step", -100)
    return out

def apply_block_perturb(S, labels, cx, cy, w, h, mode="rand", rng=None):
    """Return a perturbed copy of S."""
    import numpy as np
    H, W = S.shape
    x0 = max(0, cx - w//2); x1 = min(W, x0 + w)
    y0 = max(0, cy - h//2); y1 = min(H, y0 + h)
    P = S.copy()
    b = labels.index(BLACK); widx = labels.index(WHITE)
    rng = np.random.default_rng() if rng is None else rng

    if mode == "rand":
        P[y0:y1, x0:x1] = rng.integers(0, len(labels), size=(y1-y0, x1-x0))
    elif mode == "flipbw":
        blk = (P[y0:y1, x0:x1] == b)
        wht = (P[y0:y1, x0:x1] == widx)
        P[y0:y1, x0:x1][blk] = widx
        P[y0:y1, x0:x1][wht] = b
    elif mode == "setwhite":
        P[y0:y1, x0:x1] = widx
    elif mode == "setblack":
        P[y0:y1, x0:x1] = b
    return P

# ---- Cycle detection ----
def detect_cycle(states):
    """Return dict with preperiod/period if a repeated state is found; else None."""
    seen = {}
    for t, S in enumerate(states):
        h = hashlib.blake2b(S.tobytes(), digest_size=16).digest()
        if h in seen:
            t1 = seen[h]; t2 = t
            return {"first_seen": t1, "repeat_at": t2,
                    "preperiod": t1, "period": t2 - t1}
        seen[h] = t
    return None

# ---- Ring radius tracking (black) ----
BLACK, WHITE = "gru", "mex"

def cheb_D(H, W):
    I, J = np.indices((H, W))
    return np.minimum.reduce([I, J, (H-1)-I, (W-1)-J])

def ring_radius_series(states, labels, black_label=BLACK):
    black_idx = labels.index(black_label)
    H, W = states[0].shape
    D = cheb_D(H, W)
    maxd = int(D.max())
    radii = []
    for S in states:
        # mean black per distance; argmax
        means = [(d, (S[D==d] == black_idx).mean()) for d in range(maxd+1)]
        radii.append(int(max(means, key=lambda x: x[1])[0]))
    return np.array(radii)

def plot_ring_radius_series(radii, out_png: Path, title="Ring radius over time"):
    plt.figure(figsize=(10,3.5))
    plt.plot(np.arange(len(radii)), radii, marker='.', linewidth=0.8)
    plt.xlabel("Step"); plt.ylabel("radius (cells)"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[saved] {out_png}")

def plot_ring_radius_hist(radii, out_png: Path, title="Ring radius histogram"):
    plt.figure(figsize=(6,4))
    plt.hist(radii, bins=np.arange(radii.min()-0.5, radii.max()+1.5, 1))
    plt.xlabel("radius (cells)"); plt.ylabel("count"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[saved] {out_png}")

def learn_rule_wrap_ready(S, T, r=1):
    """Learn rule using *wrapped* keys so we can later simulate on a torus."""
    H, W = S.shape
    rule = {}
    conflicts = 0
    for i in range(H):
        for j in range(W):
            k = neigh_key_wrap(S, i, j, r=r)
            y = int(T[i, j])
            if k in rule and rule[k] != y: conflicts += 1
            else: rule[k] = y
    return rule, conflicts

def simulate_wrap(initial, rule, r=1, steps=50):
    states = [initial.copy()]
    for _ in range(steps):
        states.append(apply_rule_wrap(states[-1], rule, r=r))
    return states

# --- Phase rulebooks (exact) -----------------------------------------------

def region_of(i, j, H, W):
    if (i in (0, H-1)) and (j in (0, W-1)): return "corner"
    if (i in (0, H-1)) or (j in (0, W-1)):  return "edge"
    return "interior"

def _scope_fn(split, region, H, W):
    if split == "none" or region == "all":
        return lambda i, j: True
    if split == "edge":
        if region == "interior":
            return lambda i, j: (i not in (0, H-1) and j not in (0, W-1))
        if region == "edge":
            return lambda i, j: (i in (0, H-1) or j in (0, W-1))
    if split == "corner_edge":
        if region == "corner":
            return lambda i, j: (i in (0, H-1) and j in (0, W-1))
        if region == "edge":
            return lambda i, j: ((i in (0, H-1) or j in (0, W-1)) and not (i in (0, H-1) and j in (0, W-1)))
        if region == "interior":
            return lambda i, j: (i not in (0, H-1) and j not in (0, W-1))
    return lambda i, j: True

def build_phase_rulebooks(states, r=1, k=2, split="none"):
    """
    Build phase-conditioned rulebooks from a single run (use run_0 for coherence).
    Returns dict: {'meta': {...}, 'rules': {phase: {region: {neigh_key: out_label}}}}
    split in {'none','edge','corner_edge'}
    """
    H, W = states[0].shape
    regions = ["all"] if split == "none" else (["interior","edge"] if split=="edge" else ["interior","edge","corner"])
    rules = {phase: {reg: {} for reg in regions} for phase in range(k)}
    conflicts = 0

    for t in range(len(states)-1):
        phase = t % k
        S, T = states[t], states[t+1]
        for i in range(H):
            for j in range(W):
                for reg in regions:
                    if split != "none":
                        if not _scope_fn(split, reg, H, W)(i, j):
                            continue
                    key = neigh_key(S, i, j, r=r)  # includes OOB=-1 in your neigh_key
                    y = int(T[i, j])
                    book = rules[phase][reg]
                    if key in book and book[key] != y:
                        conflicts += 1
                    else:
                        book[key] = y
    return {
        "meta": {"radius": r, "k": k, "split": split, "H": H, "W": W, "conflicts": conflicts},
        "rules": rules
    }

def export_rulebooks(rb, outdir: Path, name="rulebooks"):
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{name}.json"
    import json
    # convert keys to json (already strings); values are ints
    p.write_text(json.dumps(rb, indent=2))
    print(f"[saved] {p}")

def load_rulebooks(path: Path):
    import json
    rb = json.loads(Path(path).read_text())
    return rb

def simulate_rulebook(initial, rb, steps=100, fallback="copy"):
    """
    Roll forward using exact phase rulebooks.
      initial: H×W int array
      rb: dict from build/load_rulebooks
      steps: number of steps to simulate
      fallback: behavior when key missing: 'copy' (default) or 'error'
    Returns list of states [S0, S1, ..., Ssteps]
    """
    r = rb["meta"]["radius"]; k = rb["meta"]["k"]; split = rb["meta"]["split"]
    H, W = initial.shape
    regions = ["all"] if split == "none" else (["interior","edge"] if split=="edge" else ["interior","edge","corner"])
    states = [initial.copy()]
    for t in range(steps):
        phase = t % k
        S = states[-1]
        T = np.empty_like(S)
        # choose which region book applies per cell
        for i in range(H):
            for j in range(W):
                reg = "all"
                if split != "none":
                    reg = region_of(i, j, H, W)
                key = neigh_key(S, i, j, r=r)
                book = rb["rules"][str(phase)][reg] if isinstance(rb["rules"], dict) and str(phase) in rb["rules"] else rb["rules"][phase][reg]
                if key in book:
                    T[i, j] = book[key]
                else:
                    if fallback == "copy":
                        T[i, j] = S[i, j]
                    else:
                        raise KeyError(f"Unseen neighborhood at t={t}, (i,j)=({i},{j}), phase={phase}, region={reg}")
        states.append(T)
    return states

# --------------- CLI runners ---------------

def run_analysis(run_dirs, outdir: Path, radius=1, tests=None, period_scan_mode="",
                 train_clf=False, clf_period=None, do_cycle=False, do_ring=False, do_torus=False,
                 do_autocorr=False, do_near_cycle=False, near_cycle_maxlag=64, near_cycle_eps=0.01,
                 perturb_spec="", simulate_spec="", sim_steps=0, sim_save_every=0, sim_png=False, sim_seed_colors="label_colors.json",
                 build_rulebook=False, rb_k=None, rb_split="none", rb_name="rulebooks",
                 simulate_rulebook_from="", rb_steps=0, rb_fallback="copy", rb_png=False, rb_save_every=0, rb_colors="label_colors.json",
                 window: str | None = None, scope_mask: np.ndarray | None = None, scope_desc: str | None = None):

    tests = set((tests or "").split(",")) if tests else set()
    all_states, labels, label_to_idx, per_run = load_runs_encoded(run_dirs, window=window)
    ensure_dir(outdir)

    if not per_run or not per_run[0]:
        raise ValueError("No data available after applying window; unable to analyze")

    if scope_mask is not None:
        scope_mask = scope_mask.astype(bool)
        mask_shape = scope_mask.shape
        for run in per_run:
            if not run:
                continue
            if run[0].shape != mask_shape:
                raise ValueError(f"Scope mask shape {mask_shape} does not match run grid shape {run[0].shape}.")
        active = int(scope_mask.sum())
        total = int(scope_mask.size)
        if active == 0:
            raise ValueError("Scope mask selects zero cells; nothing to analyze.")
        pct = (active / total) * 100.0
        desc = scope_desc or "custom mask"
        print(f"[scope] applying mask '{desc}' -> {active}/{total} cells ({pct:.2f}%)")

    # Coverage (aggregate)
    seen = neighborhood_counts(all_states, r=radius, scope_mask=scope_mask)
    uniq_count = len(seen)
    print(f"[aggregate] unique neighborhoods (r={radius}): {uniq_count}")

    # Global conflicts (aggregate)
    global_rule, global_conf = aggregate_conflicts(all_states, r=radius, scope_mask=scope_mask)
    print(f"[aggregate] global conflicts: {global_conf}, rule size: {len(global_rule)}")
    summary_payload = {
        "unique_neighborhoods_r": radius,
        "unique_count": uniq_count,
        "global_conflicts": global_conf,
        "rule_size": len(global_rule),
        "window": window,
    }
    if scope_mask is not None:
        summary_payload["scope"] = {
            "description": scope_desc or "",
            "active_cells": int(scope_mask.sum()),
            "total_cells": int(scope_mask.size),
        }
    save_json(summary_payload, outdir / "summary.json")

    # Plots on first run (for a concrete picture)
    if per_run:
        from .viz import load_label_colors, plot_sequence
        colors = load_label_colors("label_colors.json")
        plot_mismatch_curve(per_run[0], r=radius, outdir=outdir, prefix="", scope_mask=scope_mask)
        plot_label_histogram(per_run[0], labels, colors, outdir=outdir, prefix="")
        if scope_mask is not None:
            slug = re.sub(r"[^A-Za-z0-9]+", "_", scope_desc or "scoped").strip("_") or "scoped"
            plot_label_histogram(
                per_run[0], labels, colors, outdir=outdir, prefix="",
                scope_mask=scope_mask, filename_suffix=f"_{slug}"
            )

    # Consistency across runs
    try:
        df_cons = consistency_summary(per_run, labels)
        df_cons.to_csv(outdir / "consistency_summary.csv", index=False)
        print(f"[saved] {outdir/'consistency_summary.csv'}")
    except Exception as e:
        print("[warn] consistency summary failed:", e)

    # Conflict inspection & export
    if "conflicts" in tests or not tests:
        examples = extract_conflict_examples(all_states, labels, r=radius, max_examples=100, scope_mask=scope_mask)
        save_json({"count": len(examples), "examples": examples}, outdir / "conflicts_examples.json")
        # also CSV (first neighborhood flattened)
        rows = [["t","i","j","radius","outputs","neighborhood_flat"]]
        for e in examples:
            flat = " ".join([c for row in e["neighborhood_labels"] for c in row])
            rows.append([e["t"], e["i"], e["j"], e["radius"],
                         " ".join(e["outputs_labels"]), flat])
        save_csv(rows, outdir / "conflicts_examples.csv")
        print(f"[saved] {outdir/'conflicts_examples.json'}, {outdir/'conflicts_examples.csv'}")

    # Permutation invariance test
    if "permutation" in tests or not tests:
        perm_res = permutation_invariance_test(all_states, labels, trials=20, r=radius, scope_mask=scope_mask)
        save_json({"results": perm_res}, outdir / "permutation_test.json")
        rows = [["step","baseline_mismatches","trials","worse","same","better"]]
        for rrow in perm_res:
            rows.append([rrow["step"], rrow["baseline_mismatches"], rrow["trials"], rrow["worse"], rrow["same"], rrow["better"]])
        save_csv(rows, outdir / "permutation_test.csv")
        print(f"[saved] {outdir/'permutation_test.json'}, {outdir/'permutation_test.csv'}")

    # Totalistic vs positional
    if "totalistic" in tests or not tests:
        cmp_rows = compare_totalistic_vs_positional(all_states, r=radius, scope_mask=scope_mask)
        save_csv(cmp_rows, outdir / "totalistic_vs_positional.csv")
        print(f"[saved] {outdir/'totalistic_vs_positional.csv'}")

    if period_scan_mode:
        raw_modes = [m.strip() for m in period_scan_mode.split(",") if m.strip()]
        scan_specs: list[tuple[str, str, int]] = []
        for raw in raw_modes:
            split_name = raw or "none"
            max_k_local = 32
            if ":" in raw:
                split_part, max_part = raw.split(":", 1)
                split_name = split_part or "none"
                try:
                    max_k_local = int(max_part)
                except ValueError:
                    print(f"[warn] Invalid period-scan max_k '{max_part}' for '{raw}'; using 32")
                    max_k_local = 32
            scan_specs.append((raw, split_name, max_k_local))

        for raw_mode, split_name, max_k_local in scan_specs:
            rows = period_scan(per_run, r=radius, max_k=max_k_local, split=split_name, scope_mask=scope_mask)
            out_path = outdir / f"period_scan_{raw_mode}.csv"
            save_csv(rows, out_path)
            print(f"[saved] {out_path}")
        try:
            summarise_period_scan(outdir, [spec[0] for spec in scan_specs])
        except Exception as e:
            print(f"[warn] period scan summary failed: {e}")

    if train_clf:
        # pick a k: use provided clf_period, or guess small (e.g. 2 or from your period_scan)
        k = int(clf_period) if clf_period else 2
        print(f"[clf] training logistic regression with k={k}, r={radius}")
        clf, meta = train_classifier(all_states, labels, r=radius, k=k, max_steps=200, C=1.0)
        export_classifier(clf, meta, outdir)

        # quick smoke test: simulate 30 steps from the first state of run_0 and save PNGs
        init = per_run[0][0]  # first frame of first run
        sim_states = simulate_with_classifier(init, clf, meta, steps=30)
        try:
            from .viz import load_label_colors, plot_sequence
            colors = load_label_colors("label_colors.json")
            plot_sequence(sim_states, labels, colors, steps=list(range(0,31,5)))
        except Exception as e:
            print("[warn] plotting failed (viz.py not present or colors missing):", e)

    # ---------------- Simulate-only rollout ----------------
    if simulate_spec:
        # load classifier + meta
        try:
            import joblib
            clf = joblib.load(outdir / "classifier.joblib")
            meta = json.loads((outdir / "classifier_meta.json").read_text())
        except Exception as e:
            print("[error] simulate requires classifier.joblib + classifier_meta.json in reports/:", e)
            return

        # decide initial
        init = choose_initial_state(simulate_spec, per_run, labels, label_to_idx)
        steps = int(sim_steps) if sim_steps else 0
        if steps <= 0:
            print("[warn] --simulate provided but --sim-steps not set (>0). Nothing to do.")
            return

        sim_states = simulate_with_classifier(init, clf, meta, steps=steps)

        # save JSONs
        save_sequence_json(sim_states, labels, outdir, prefix="simulate_", save_every=int(sim_save_every))

        # optional PNG panel
        if sim_png:
            save_sequence_png(sim_states, labels, outdir, prefix="simulate_", colors_path=sim_seed_colors)

        # also dump a fractions CSV for the rollout
        try:
            import numpy as np, pandas as pd
            H, W = sim_states[0].shape
            total = H*W
            rows = []
            for t, S in enumerate(sim_states):
                vals, counts = np.unique(S, return_counts=True)
                freq = dict(zip(vals, counts))
                for idx, lab in enumerate(labels):
                    rows.append([t, lab, freq.get(idx, 0)/total])
            import csv
            with open(outdir / "simulate_fractions.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["step","label","fraction"]); w.writerows(rows)
            print(f"[saved] {outdir/'simulate_fractions.csv'}")
        except Exception as e:
            print("[warn] could not write simulate_fractions.csv:", e)

    # -------- Build and export rulebooks (exact) --------
    if build_rulebook:
        if not per_run:
            print("[error] need at least one run to build rulebooks")
            return
        k = int(rb_k) if rb_k else 2
        rb = build_phase_rulebooks(per_run[0], r=radius, k=k, split=rb_split)
        export_rulebooks(rb, outdir, name=rb_name)
        print(f"[rulebooks] conflicts while building: {rb['meta']['conflicts']}")

    # -------- Simulate using rulebooks --------
    if simulate_rulebook_from:
        # load rulebooks (prefer the just-saved one if present)
        rb_path = outdir / f"{rb_name}.json"
        if not rb_path.exists():
            print(f"[error] missing {rb_path}. Build with --build-rulebook first or set --rb-name to existing file.")
            return
        rb = load_rulebooks(rb_path)

        # pick initial
        init = choose_initial_state(simulate_rulebook_from, per_run, labels, label_to_idx)
        steps = int(rb_steps) if rb_steps else 0
        if steps <= 0:
            print("[warn] --simulate-rulebook provided but --rb-steps not set (>0). Nothing to do.")
            return

        sim_states = simulate_rulebook(init, rb, steps=steps, fallback=rb_fallback)

        # save JSON snapshots
        save_sequence_json(sim_states, labels, outdir, prefix="rb_sim_", save_every=int(rb_save_every))
        # optional PNG montage
        if rb_png:
            save_sequence_png(sim_states, labels, outdir, prefix="rb_sim_", colors_path=rb_colors)

        # fractions CSV for the rollout
        try:
            H, W = sim_states[0].shape
            total = H*W
            rows = []
            for t, S in enumerate(sim_states):
                vals, counts = np.unique(S, return_counts=True)
                freq = dict(zip(vals, counts))
                for idx, lab in enumerate(labels):
                    rows.append([t, lab, freq.get(idx, 0)/total])
            import csv
            with open(outdir / "rb_simulate_fractions.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["step","label","fraction"]); w.writerows(rows)
            print(f"[saved] {outdir/'rb_simulate_fractions.csv'}")
        except Exception as e:
            print("[warn] could not write rb_simulate_fractions.csv:", e)


# --- simulate-only helpers (surrogate rollout) ---

def encode_by_global_labels(grid_labels_2d, label_to_idx):
    """Take a raw 2D grid of label strings and encode to ints with a global map."""
    import numpy as np
    H, W = len(grid_labels_2d), len(grid_labels_2d[0])
    S = np.empty((H, W), dtype=np.int32)
    for i in range(H):
        for j in range(W):
            S[i, j] = label_to_idx[grid_labels_2d[i][j]]
    return S

def decode_by_global_labels(S_int, labels):
    """Decode an int grid back to label strings."""
    H, W = S_int.shape
    return [[labels[int(S_int[i, j])] for j in range(W)] for i in range(H)]

def load_seed_from_file(path, labels, label_to_idx):
    """Load a JSON seed (like the captured step_XXXX.json) and encode by global labels."""
    from pathlib import Path
    import json
    grid = json.loads(Path(path).read_text())
    return encode_by_global_labels(grid, label_to_idx)

def choose_initial_state(sim_spec, per_run_states, labels, label_to_idx):
    """
    sim_spec formats:
      - 'run:idx=<k>'      → use per_run_states[0][k]
      - 'run:last'         → last state from run_0
      - 'file:<path.json>' → load a seed JSON and encode
    """
    if not sim_spec:
        # default: first frame of first run
        return per_run_states[0][0]

    if sim_spec.startswith("run:"):
        arg = sim_spec[4:]
        if arg == "last":
            return per_run_states[0][-1]
        if arg.startswith("idx="):
            k = int(arg.split("=", 1)[1])
            k = max(0, min(k, len(per_run_states[0]) - 1))
            return per_run_states[0][k]
        raise ValueError(f"Bad run spec: {sim_spec}")

    if sim_spec.startswith("file:"):
        path = sim_spec[5:]
        return load_seed_from_file(path, labels, label_to_idx)

    raise ValueError(f"Unrecognized simulate spec: {sim_spec}")

def save_sequence_json(states, labels, outdir: Path, prefix="sim_", save_every=0):
    """
    Save decoded states as JSON label grids.
    If save_every==0: saves only final state → <prefix>final.json
    If save_every>0 : saves every k steps and final.
    """
    import json
    outdir.mkdir(parents=True, exist_ok=True)
    if save_every and save_every > 0:
        for t, S in enumerate(states):
            if t % save_every == 0:
                path = outdir / f"{prefix}t{t:04d}.json"
                path.write_text(json.dumps(decode_by_global_labels(S, labels)))
        # always save final too (even if it aligned with save_every)
        path = outdir / f"{prefix}final.json"
        path.write_text(json.dumps(decode_by_global_labels(states[-1], labels)))
        return

    # only final
    path = outdir / f"{prefix}final.json"
    path.write_text(json.dumps(decode_by_global_labels(states[-1], labels)))

def save_sequence_png(states, labels, outdir: Path, prefix="sim_", steps_to_plot=None, colors_path="label_colors.json"):
    """
    Save a grid of snapshots as PNG using viz.py if available.
    If steps_to_plot is None, picks ~15 evenly spaced snapshots including 0 and final.
    """
    try:
        from viz import load_label_colors, plot_sequence
        import numpy as np
        colors = load_label_colors(colors_path)
        if steps_to_plot is None:
            N = min(15, len(states))
            steps_to_plot = np.linspace(0, len(states) - 1, N, dtype=int).tolist()
        outdir.mkdir(parents=True, exist_ok=True)
        # plot_sequence shows the figure; we want to save instead:
        # reimplement a small saver
        from matplotlib.colors import ListedColormap
        import matplotlib.pyplot as plt
        cmap = ListedColormap([colors.get(lab, "#808080") for lab in labels], name="bbx")
        cols = 5
        rows = -(-len(steps_to_plot)//cols)
        plt.figure(figsize=(12, 8))
        for k, t in enumerate(steps_to_plot):
            ax = plt.subplot(rows, cols, k+1)
            ax.imshow(states[t], cmap=cmap, interpolation="nearest", vmin=0, vmax=len(labels)-1)
            ax.set_title(f"t={t}", fontsize=9); ax.axis("off")
        plt.tight_layout()
        p = outdir / f"{prefix}snapshots.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[saved] {p}")
    except Exception as e:
        print("[warn] PNG sequence save skipped (viz.py/colors missing):", e)


    # --- Shape diagnostics (ring/border evidence) ---
    if "shape" in tests or not tests:
        # Use the first run for clean geometry pictures (you can aggregate later)
        if per_run:
            states = per_run[0]
            H, W = states[0].shape

            # 1) Radial profile
            df_rad = black_radial_profile(states, labels, black_label=BLACK)
            save_csv([["step","d","black_fraction"]] + df_rad.values.tolist(),
                     outdir / "shape_radial_black.csv")
            plot_black_radial_profile(df_rad,
                                      out_png=outdir / "shape_radial_black.png",
                                      title=f"Black vs distance (run_0)")

            # 2) Row/Col peaks over time
            df_rc = black_by_row_col(states, labels, black_label=BLACK)
            save_csv([["step","row_max","row_argmax","col_max","col_argmax"]] + df_rc.values.tolist(),
                     outdir / "shape_rowcol_black.csv")
            plot_row_col_traces(df_rc, H, W,
                                out_png=outdir / "shape_rowcol_black.png",
                                title="Row/Col of max black fraction (run_0)")

            # 3) Band vs interior fractions (tweak d0/d1 if your ring sits deeper)
            d0, d1 = 2, 3
            df_band = band_vs_interior_fractions(states, labels, d0=d0, d1=d1,
                                                 black_label=BLACK, white_label=WHITE)
            df_band.to_csv(outdir / f"shape_band_d{d0}-{d1}.csv", index=False)
            plot_band_vs_interior(df_band,
                                  out_png=outdir / f"shape_band_d{d0}-{d1}.png",
                                  d0=d0, d1=d1,
                                  title=f"Band d∈[{d0},{d1}] vs interior — black/white (run_0)")
            print("[saved] shape diagnostics CSVs/PNGs")

    # ---- Cycle detection + ring tracking on first run ----
    if per_run:
        states0 = per_run[0]
        # Cycle detection
        if do_cycle:
            cyc = detect_cycle(states0)
            save_json({"cycle": cyc}, outdir / "cycle_report_run0.json")
            print("[cycle] run_0:", cyc)

        # Ring tracking
        if do_ring:
            radii = ring_radius_series(states0, labels, black_label=BLACK)
            pd.Series(radii).to_csv(outdir / "ring_radius_run0.csv", index=False, header=["radius"])
            plot_ring_radius_series(radii, out_png=outdir / "ring_radius_run0.png",
                                    title="Ring radius over time — run_0")
            plot_ring_radius_hist(radii, out_png=outdir / "ring_radius_hist_run0.png",
                                  title="Ring radius histogram — run_0")

    # ---- Torus (wrap) test: learn on early transition and simulate on a crop with wrap ----
    if do_torus and per_run:
        states0 = per_run[0]
        S, T = states0[0], states0[1]  # learn from first transition (you can average later)
        rb_wrap, conf = learn_rule_wrap_ready(S, T, r=radius)
        print(f"[torus] learned wrap-ready rule, size={len(rb_wrap)}, conflicts={conf}")

        # start from a central crop to avoid edges
        H, W = S.shape
        a, b = 1, H-1  # drop 1-cell border
        init_crop = S[a:b, a:b].copy()
        sim = simulate_wrap(init_crop, rb_wrap, r=radius, steps=100)

        # summarize: black fraction over time on torus
        idx_black = labels.index(BLACK)
        blk = [float((X == idx_black).mean()) for X in sim]
        pd.DataFrame({"step": np.arange(len(blk)), "black_frac": blk}) \
          .to_csv(outdir / "torus_black_frac.csv", index=False)

        # quick plot
        plt.figure(figsize=(8,3))
        plt.plot(blk, marker='.')
        plt.xlabel("step"); plt.ylabel("black fraction"); plt.title("Torus sim (wrap) — black fraction")
        plt.tight_layout(); plt.savefig(outdir / "torus_black_frac.png", dpi=150, bbox_inches="tight"); plt.close()
        print("[saved] torus_black_frac.csv/.png")

# ---------------- Autocorrelation ----------------
    if do_autocorr and per_run:
        import numpy as np
        run = per_run[0]
        fr = fractions_time_series(run, labels)
        ring = ring_radius_series(run, labels)

        # AC for black & white fractions and ring radius
        ac_black = autocorr_1d(fr.get(BLACK, np.zeros(len(run))), max_lag=128)
        ac_white = autocorr_1d(fr.get(WHITE, np.zeros(len(run))), max_lag=128)
        ac_ring  = autocorr_1d(ring, max_lag=128)

        save_json({
            "lags": list(range(len(ac_black))),
            "ac_black": ac_black.tolist(),
            "ac_white": ac_white.tolist(),
            "ac_ring_radius": ac_ring.tolist()
        }, outdir / "autocorr_run0.json")
        print(f"[saved] {outdir/'autocorr_run0.json'}")

        # quick PNGs
        for name, ac in [("black", ac_black), ("white", ac_white), ("ring", ac_ring)]:
            plt.figure(); plt.plot(range(len(ac)), ac, marker='.')
            plt.xlabel("lag"); plt.ylabel("autocorr"); plt.title(f"Autocorr ({name})")
            p = outdir / f"autocorr_{name}.png"
            plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
            print(f"[saved] {p}")

    # ---------------- Near-cycle detection ----------------
    if do_near_cycle and per_run:
        run = per_run[0]
        import numpy as np
        # collapse to B/W/Other to ignore micro-color shuffles
        collapsed = [collapse_bw_other(S, labels) for S in run]
        hits = []
        mins = []
        for lag in range(1, near_cycle_maxlag+1):
            ds = []
            for t in range(lag, len(collapsed)):
                d = hamming_fraction(collapsed[t], collapsed[t-lag])
                ds.append(d)
            if ds:
                dmin = float(np.min(ds))
                mins.append({"lag": lag, "min_mismatch_frac": dmin})
                if dmin <= near_cycle_eps:
                    hits.append({"lag": lag, "best_mismatch_frac": dmin})
        save_json({"threshold": near_cycle_eps, "maxlag": near_cycle_maxlag,
                   "hits": hits, "min_by_lag": mins}, outdir / "near_cycle_report.json")
        rows = [["lag","min_mismatch_frac"]] + [[m["lag"], m["min_mismatch_frac"]] for m in mins]
        save_csv(rows, outdir / "near_cycle_min_by_lag.csv")
        print(f"[saved] {outdir/'near_cycle_report.json'}, {outdir/'near_cycle_min_by_lag.csv'}")

        # plot min mismatch vs lag
        if mins:
            lags = [m["lag"] for m in mins]; vals = [m["min_mismatch_frac"] for m in mins]
            plt.figure(); plt.plot(lags, vals, marker='.')
            plt.axhline(near_cycle_eps, color='r', linestyle='--', label=f"threshold={near_cycle_eps}")
            plt.xlabel("lag"); plt.ylabel("min mismatch (B/W/Other)"); plt.legend()
            p = outdir / "near_cycle_min_by_lag.png"
            plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
            print(f"[saved] {p}")

    # ---------------- Perturbation experiment ----------------
    if perturb_spec:
        spec = parse_perturb_spec(perturb_spec)
        if not per_run:
            print("[warn] no run available for perturbation")
        else:
            # need classifier + meta (exported earlier with --train-clf)
            try:
                import joblib
                clf = joblib.load(outdir / "classifier.joblib")
                meta = json.loads((outdir / "classifier_meta.json").read_text())
            except Exception as e:
                print("[error] perturbation requires classifier.joblib + classifier_meta.json in reports/:", e)
                return

            run = per_run[0]
            base_step = spec.get("base_step", -100)
            base_idx = base_step if base_step >= 0 else (len(run) + base_step)
            base_idx = max(0, min(base_idx, len(run)-1))
            base = run[base_idx]

            # make perturbed initial
            H, W = base.shape
            cx = spec.get("cx", W//2)
            cy = spec.get("cy", H//2)
            w  = spec.get("w",  5)
            h  = spec.get("h",  5)
            mode = spec.get("mode", "rand")
            steps = spec.get("steps", 100)

            pert0 = apply_block_perturb(base, labels, cx, cy, w, h, mode=mode)

            # simulate both trajectories (reference vs perturbed)
            ref_states  = simulate_with_classifier(base, clf, meta, steps=steps)
            pert_states = simulate_with_classifier(pert0, clf, meta, steps=steps)

            # metrics: black/white fractions and ring radius diffs vs reference
            fr_ref = fractions_time_series(ref_states, labels)
            fr_pt  = fractions_time_series(pert_states, labels)
            ring_ref = ring_radius_series(ref_states, labels)
            ring_pt  = ring_radius_series(pert_states, labels)

            rows = [["t","black_ref","black_pert","white_ref","white_pert","ring_ref","ring_pert",
                     "d_black","d_white","d_ring"]]
            for t in range(len(ref_states)):
                b_ref = fr_ref[BLACK][t]; b_pt = fr_pt[BLACK][t]
                w_ref = fr_ref[WHITE][t]; w_pt = fr_pt[WHITE][t]
                rr = int(ring_ref[t]); rp = int(ring_pt[t])
                rows.append([t, b_ref, b_pt, w_ref, w_pt, rr, rp,
                             b_pt-b_ref, w_pt-w_ref, rp-rr])

            save_csv(rows, outdir / "perturb_recovery.csv")
            print(f"[saved] {outdir/'perturb_recovery.csv'}")

            # quick plots
            # (a) delta black/white
            ts = [r[0] for r in rows[1:]]
            dB = [r[7] for r in rows[1:]]
            dW = [r[8] for r in rows[1:]]
            plt.figure(); plt.plot(ts, dB, label="Δ black"); plt.plot(ts, dW, label="Δ white")
            plt.axhline(0, color='k', linewidth=0.5)
            plt.xlabel("t"); plt.ylabel("fraction diff (pert - ref)"); plt.legend()
            p = outdir / "perturb_recovery_dBW.png"
            plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"[saved] {p}")

            # (b) delta ring radius
            dR = [r[9] for r in rows[1:]]
            plt.figure(); plt.plot(ts, dR, label="Δ ring radius")
            plt.axhline(0, color='k', linewidth=0.5)
            plt.xlabel("t"); plt.ylabel("cells"); plt.title("Recovery of ring radius")
            p = outdir / "perturb_recovery_dRing.png"
            plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close(); print(f"[saved] {p}")

    print("[done] analyze.py finished.")

def features_for_cell(S, i, j, r, H, W, L, k_phase=None, parity=None, region=None):
    """
    Build features for a single cell:
      - one-hot for each of the (2r+1)^2 positions with L+1 values (labels 0..L-1 plus OOB as L)
      - center label one-hot (L)
      - k_phase (if provided): small one-hot (up to, say, 16)
      - parity: (i+j) mod 2 as binary
      - region: one-hot of {'interior','edge','corner'}
    """
    size = 2*r + 1
    oob_id = L
    # positional one-hot
    pos_oh = []
    for di in range(-r, r+1):
        for dj in range(-r, r+1):
            ii, jj = i+di, j+dj
            v = oob_id
            if 0 <= ii < H and 0 <= jj < W: v = int(S[ii, jj])
            # one-hot expand to length L+1
            oh = [0]*(L+1); oh[v] = 1
            pos_oh.extend(oh)

    # center label one-hot (L)
    center = int(S[i, j])
    center_oh = [0]*L; center_oh[center] = 1

    # phase (up to 16); we’ll cap at 16 one-hot for safety
    phase_oh = [0]*16
    if k_phase is not None:
        phase_oh[k_phase % 16] = 1

    # parity bit
    par = [(i + j) & 1] if parity is not None else []

    # region one-hot (3)
    reg_oh = [0,0,0]
    if region == "interior": reg_oh = [1,0,0]
    elif region == "edge":   reg_oh = [0,1,0]
    elif region == "corner": reg_oh = [0,0,1]

    return pos_oh + center_oh + phase_oh + par + reg_oh

def region_of(i, j, H, W):
    if (i in (0,H-1)) and (j in (0,W-1)): return "corner"
    if (i in (0,H-1)) or (j in (0,W-1)):  return "edge"
    return "interior"

def build_dataset(states, r=1, k=None, L=None, use_parity=True, use_region=True, use_phase=True, max_steps=200):
    """
    Build X,y across first max_steps transitions.
    k: time period; if provided, include phase features and train a *single* classifier.
    """
    X, y = [], []
    L = int(L)
    TMAX = min(max_steps, len(states)-1)
    H, W = states[0].shape
    for t in range(TMAX):
        S, T = states[t], states[t+1]
        phase = (t % k) if (use_phase and k) else None
        for i in range(H):
            for j in range(W):
                reg = region_of(i, j, H, W) if use_region else None
                x = features_for_cell(S, i, j, r, H, W, L,
                                      k_phase=phase if use_phase else None,
                                      parity=((i+j)&1) if use_parity else None,
                                      region=reg)
                X.append(x)
                y.append(int(T[i, j]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def train_classifier(all_states, labels, r=1, k=2, max_steps=200, C=1.0):
    """
    Train multinomial logistic regression on positional features + center label + phase + parity + region.
    Returns the fitted model and metadata needed for simulation.
    """
    L = len(labels)
    X, y = build_dataset(all_states, r=r, k=k, L=L, use_parity=True, use_region=True, use_phase=True, max_steps=max_steps)
    clf = LogisticRegression(
        penalty="l2", C=C, solver="lbfgs", multi_class="multinomial", max_iter=200
    )
    clf.fit(X, y)
    meta = {
        "labels": labels,
        "radius": r,
        "period_k": k,
        "L": L,
        "feature_schema": {
            "pos_block": (2*r+1)**2 * (L+1),
            "center_block": L,
            "phase_block": 16,
            "parity_block": 1,
            "region_block": 3
        }
    }
    return clf, meta

def export_classifier(clf, meta, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, outdir / "classifier.joblib")
    save_json(meta, outdir / "classifier_meta.json")
    print(f"[saved] {outdir/'classifier.joblib'}, {outdir/'classifier_meta.json'}")

def simulate_with_classifier(initial, clf, meta, steps=50):
    """Run the learned classifier forward for N steps. Returns list of states (including initial)."""
    states = [initial.copy()]
    r = meta["radius"]; L = meta["L"]; k = meta["period_k"]
    H, W = initial.shape
    for t in range(steps):
        S = states[-1]
        phase = (t % k)
        Tnext = np.empty_like(S)
        for i in range(H):
            for j in range(W):
                reg = region_of(i, j, H, W)
                x = np.array([features_for_cell(S, i, j, r, H, W, L,
                                                k_phase=phase, parity=((i+j)&1),
                                                region=reg)], dtype=np.float32)
                pred = clf.predict(x)[0]
                Tnext[i, j] = pred
        states.append(Tnext)
    return states

def consistency_summary(per_run_states, labels):
    """Return a dict of per-run stats for easy comparison."""
    idx_black = labels.index(BLACK)
    idx_white = labels.index(WHITE)
    rows = []
    for ridx, states in enumerate(per_run_states):
        # global fractions (last 200 steps if available)
        tail = states[-200:] if len(states) > 200 else states
        blk = [float((S == idx_black).mean()) for S in tail]
        wht = [float((S == idx_white).mean()) for S in tail]
        # ring radius
        radii = ring_radius_series(tail, labels, black_label=BLACK)
        rows.append({
            "run": ridx,
            "steps": len(states),
            "black_mean": float(np.mean(blk)), "black_std": float(np.std(blk)),
            "white_mean": float(np.mean(wht)), "white_std": float(np.std(wht)),
            "ring_mode": int(pd.Series(radii).mode().iloc[0]),
            "ring_mean": float(np.mean(radii)), "ring_std": float(np.std(radii)),
        })
    return pd.DataFrame(rows)

# --------------- main ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--tests", default="")
    ap.add_argument("--period-scan", default="")
    ap.add_argument("--train-clf", action="store_true")
    ap.add_argument("--clf-period", default="")
    ap.add_argument("--do-cycle", default="")
    ap.add_argument("--do-ring", default="")
    ap.add_argument("--do-torus", defaults="")
    ap.add_argument("--autocorr", action="store_true", help="Compute autocorrelation for black/white fractions and ring radius")
    ap.add_argument("--near-cycle", action="store_true", help="Near-cycle detection on B/W/Other-collapsed states")
    ap.add_argument("--near-cycle-maxlag", type=int, default=64, help="Max lag to scan for near-cycles (default 64)")
    ap.add_argument("--near-cycle-eps", type=float, default=0.01, help="Mismatch fraction threshold for a near-cycle hit (default 1%)")
    ap.add_argument("--perturb", default="", help="Perturbation spec, e.g. 'block:cx=10,cy=10,w=5,h=5,mode=rand,steps=100,base_step=-100'")
    args = ap.parse_args()

    outdir = ensure_dir(Path(args.out))
    run_analysis(
        args.runs, outdir,
        radius=args.radius,
        tests=args.tests,
        period_scan_mode=args["period_scan"] if isinstance(args, dict) else args.period_scan,
        train_clf=args.train_clf,
        clf_period=args.clf_period,
        do_cycle=args.do_cycle,
        do_ring=args.do_ring,
        do_torus=args.do_torus,
        do_autocorr=args.autocorr,
        do_near_cycle=args.near_cycle,
        near_cycle_maxlag=args.near_cycle_maxlag,
        near_cycle_eps=args.near_cycle_eps,
        perturb_spec=args.perturb
    )
