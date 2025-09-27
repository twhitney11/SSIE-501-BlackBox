#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
import hashlib

# ---- import local helpers from process.py ----
from .process import (
    load_run, build_label_map, encode_states,
    learn_rule, apply_rule, diff_count, neigh_key, coverage
)

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

# --------------- loading & encoding ---------------

def load_runs_encoded(run_dirs):
    """Return (all_states_encoded, labels_global, label_to_idx_global, per_run_states)"""
    per_run_raw = []
    for rd in run_dirs:
        raw = load_run(rd)
        if not raw:
            print(f"[warn] empty run: {rd}")
            continue
        per_run_raw.append(raw)

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

def period_scan(per_run_states, r=1, max_k=12, split="none"):
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

def neighborhood_counts(states, r=1, oob_val=-1):
    seen = Counter()
    for S in states:
        H, W = S.shape
        for i in range(H):
            for j in range(W):
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

def aggregate_conflicts(states, r=1):
    """Merge transitions across all consecutive pairs; count conflicts and build a global rulebook."""
    rule = {}
    conflicts = 0
    for t in range(len(states) - 1):
        S, T = states[t], states[t + 1]
        H, W = S.shape
        for i in range(H):
            for j in range(W):
                k = neigh_key(S, i, j, r=r)
                y = int(T[i, j])
                if k in rule and rule[k] != y:
                    conflicts += 1
                else:
                    rule[k] = y
    return rule, conflicts

def extract_conflict_examples(states, labels, r=1, max_examples=50):
    """Return list of dicts with neighborhood (as label grid) and outputs observed."""
    H, W = states[0].shape
    seen = {}
    examples = []
    for t in range(len(states) - 1):
        S, T = states[t], states[t + 1]
        for i in range(H):
            for j in range(W):
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

def permutation_invariance_test(states, labels, trials=20, r=1, seed=42):
    """Shuffle label ids and see if step-accuracy for 0->1 changes; repeat across early steps."""
    rng = random.Random(seed)
    L = len(labels)
    results = []
    steps_to_test = list(range(0, min(10, len(states) - 1)))  # first 10 transitions
    for t in steps_to_test:
        S, T = states[t], states[t+1]
        base_rule, _ = learn_rule(S, T, r=r)
        base_pred = apply_rule(S, base_rule, r=r)
        base_mis = diff_count(base_pred, T)
        worse = same = better = 0
        for _ in range(trials):
            perm = list(range(L)); rng.shuffle(perm)
            Sperm = np.vectorize(lambda x: perm[x])(S)
            Tperm = np.vectorize(lambda x: perm[x])(T)
            rb, _ = learn_rule(Sperm, Tperm, r=r)
            pred = apply_rule(Sperm, rb, r=r)
            mis = diff_count(pred, Tperm)
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

def learn_totalistic(S, T, r=1, oob=-1):
    H, W = S.shape
    rule = {}
    conflicts = 0
    for i in range(H):
        for j in range(W):
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

def compare_totalistic_vs_positional(states, r=1):
    """Return table of mismatches over first K transitions for both models."""
    K = min(20, len(states) - 1)
    rows = [["step", "total_cells", "mismatch_positional", "mismatch_totalistic"]]
    for t in range(K):
        S, T = states[t], states[t+1]
        rb_pos, _ = learn_rule(S, T, r=r)
        pred_pos = apply_rule(S, rb_pos, r=r)
        mis_pos = diff_count(pred_pos, T)

        rb_tot, _ = learn_totalistic(S, T, r=r)
        pred_tot = apply_totalistic(S, rb_tot, r=r)
        mis_tot = diff_count(pred_tot, T)

        rows.append([f"{t}->{t+1}", S.size, mis_pos, mis_tot])
    return rows

# --------------- plotting ---------------

def plot_mismatch_curve(states, r=1, outdir: Path = None, prefix=""):
    """Train on step t->t+1 and report exact replay mismatch for each of first K transitions."""
    K = min(100, len(states) - 1)
    xs = []
    ms = []
    for t in range(K):
        S, T = states[t], states[t+1]
        rb, _ = learn_rule(S, T, r=r)
        pred = apply_rule(S, rb, r=r)
        ms.append(diff_count(pred, T))
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

def plot_label_histogram(states, labels, outdir: Path = None, prefix=""):
    """Histogram of labels in final state vs initial state."""
    init = states[0].ravel()
    final = states[-1].ravel()
    L = len(labels)
    n_init = np.bincount(init, minlength=L)
    n_final = np.bincount(final, minlength=L)

    x = np.arange(L)
    width = 0.38
    plt.figure(figsize=(10,4))
    plt.bar(x - width/2, n_init, width, label="initial")
    plt.bar(x + width/2, n_final, width, label="final")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("count")
    plt.title("Label histogram (initial vs final)")
    plt.legend()
    plt.tight_layout()
    if outdir:
        p = outdir / f"{prefix}label_hist_init_final.png"
        plt.savefig(p, dpi=150)
        print(f"[saved] {p}")
    plt.close()

# ----- SHAPE DIAGNOSTICS (ring/border evidence) -----

BLACK, WHITE = "gru", "mex"   # adjust if your legend changes

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

# ---- Torus (wrap) simulation using learned positional rule ----
from process import neigh_key_wrap, apply_rule_wrap  # added in process.py

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

# --------------- CLI runners ---------------

def run_analysis(run_dirs, outdir: Path, radius=1, tests=None,
                 period_scan_mode="", train_clf=False, clf_period=None,
                 do_cycle=True, do_ring=True, do_torus=True):

    tests = set((tests or "").split(",")) if tests else set()
    all_states, labels, label_to_idx, per_run = load_runs_encoded(run_dirs)
    ensure_dir(outdir)

    # Coverage (aggregate)
    uniq_count, seen = coverage(all_states, r=radius)
    print(f"[aggregate] unique neighborhoods (r={radius}): {uniq_count}")

    # Global conflicts (aggregate)
    global_rule, global_conf = aggregate_conflicts(all_states, r=radius)
    print(f"[aggregate] global conflicts: {global_conf}, rule size: {len(global_rule)}")
    save_json({"unique_neighborhoods_r": radius,
               "unique_count": uniq_count,
               "global_conflicts": global_conf,
               "rule_size": len(global_rule)},
              outdir / "summary.json")

    # Plots on first run (for a concrete picture)
    if per_run:
        plot_mismatch_curve(per_run[0], r=radius, outdir=outdir, prefix="run0_")
        plot_label_histogram(per_run[0], labels, outdir=outdir, prefix="run0_")

    # Consistency across runs
    try:
        df_cons = consistency_summary(per_run, labels)
        df_cons.to_csv(outdir / "consistency_summary.csv", index=False)
        print(f"[saved] {outdir/'consistency_summary.csv'}")
    except Exception as e:
        print("[warn] consistency summary failed:", e)

    # Conflict inspection & export
    if "conflicts" in tests or not tests:
        examples = extract_conflict_examples(all_states, labels, r=radius, max_examples=100)
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
        perm_res = permutation_invariance_test(all_states, labels, trials=20, r=radius)
        save_json({"results": perm_res}, outdir / "permutation_test.json")
        rows = [["step","baseline_mismatches","trials","worse","same","better"]]
        for rrow in perm_res:
            rows.append([rrow["step"], rrow["baseline_mismatches"], rrow["trials"], rrow["worse"], rrow["same"], rrow["better"]])
        save_csv(rows, outdir / "permutation_test.csv")
        print(f"[saved] {outdir/'permutation_test.json'}, {outdir/'permutation_test.csv'}")

    # Totalistic vs positional
    if "totalistic" in tests or not tests:
        cmp_rows = compare_totalistic_vs_positional(all_states, r=radius)
        save_csv(cmp_rows, outdir / "totalistic_vs_positional.csv")
        print(f"[saved] {outdir/'totalistic_vs_positional.csv'}")

    if period_scan_mode:
        modes = period_scan_mode.split(",")
        for mode in modes:
            rows = period_scan(per_run, r=radius, max_k=12, split=mode)
            save_csv(rows, outdir / f"period_scan_{mode}.csv")
            print(f"[saved] {outdir/f'period_scan_{mode}.csv'}")

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
    args = ap.parse_args()

    outdir = ensure_dir(Path(args.out))
    run_analysis(
        args.runs, outdir,
        radius=args.radius,
        tests=args.tests,
        period_scan_mode=args["period_scan"] if isinstance(args, dict) else args.period_scan,
        train_clf=args.train_clf,
        clf_period=args.clf_period
    )

