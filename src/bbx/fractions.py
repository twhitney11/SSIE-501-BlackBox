# fractions.py
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .viz import load_label_colors
from .process import load_run_encoded

BLACK, WHITE = "gru", "mex"  # from your legend

# ----------------- core computations -----------------

def compute_color_fractions_df(states, labels):
    """states: list of H×W int arrays; returns long DF (step,label,fraction)."""
    H, W = states[0].shape
    total = H * W
    rows = []
    for t, S in enumerate(states):
        vals, counts = np.unique(S, return_counts=True)
        freq = dict(zip(vals, counts))
        for idx, lab in enumerate(labels):
            rows.append({"step": t, "label": lab, "fraction": freq.get(idx, 0) / total})
    return pd.DataFrame(rows)

def smoothed_per_label(df_long, window=7):
    """Rolling mean per label (centered)."""
    return (df_long.sort_values(["label", "step"])
                  .groupby("label", group_keys=False)
                  .apply(lambda g: g.assign(fraction=g["fraction"]
                                            .rolling(window, center=True, min_periods=1)
                                            .mean()))
            )

def region_mask(H, W):
    interior = np.ones((H, W), dtype=bool)
    interior[[0, -1], :] = False; interior[:, [0, -1]] = False
    edge = ~interior
    corner = np.zeros((H, W), dtype=bool)
    corner[0, 0] = corner[0, -1] = corner[-1, 0] = corner[-1, -1] = True
    edge_no_corner = edge & ~corner
    return interior, edge_no_corner, corner

def fractions_by_region(states, labels):
    H, W = states[0].shape
    interior, edge, corner = region_mask(H, W)
    rows = []
    for t, S in enumerate(states):
        for name, mask in [("corner", corner), ("edge", edge), ("interior", interior)]:
            vals, counts = np.unique(S[mask], return_counts=True)
            freq = dict(zip(vals, counts))
            denom = int(mask.sum()) or 1
            for idx, lab in enumerate(labels):
                rows.append({
                    "step": t, "region": name, "label": lab,
                    "fraction": freq.get(idx, 0) / denom
                })
    return pd.DataFrame(rows)

def diff_curve(states):
    return np.array([int((states[t] != states[t+1]).sum()) for t in range(len(states)-1)])

def l_inf_front_radius(S, black_idx):
    """Estimate distance (cells) of the black front from the grid edge."""
    H, W = S.shape
    rows_with = np.where((S == black_idx).any(axis=1))[0]
    cols_with = np.where((S == black_idx).any(axis=0))[0]
    if len(rows_with) == 0 or len(cols_with) == 0:
        return 0
    d_top = rows_with.min()
    d_bottom = (H - 1) - rows_with.max()
    d_left = cols_with.min()
    d_right = (W - 1) - cols_with.max()
    return min(d_top, d_bottom, d_left, d_right)

# ----------------- plotting (saved to files) -----------------

def save_plot(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")

def plot_color_fractions(df_long, colors, out_png: Path, title="Color fractions over time"):
    plt.figure(figsize=(10, 6))
    for lab, sub in df_long.groupby("label"):
        c = colors.get(lab, "#808080")
        plt.plot(sub["step"], sub["fraction"], label=lab, color=c)
    plt.xlabel("Step"); plt.ylabel("Fraction of grid"); plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    save_plot(out_png)

def plot_stacked_black_white_other(df_long, colors, out_png: Path, title="Fractions (stacked)"):
    piv = df_long.pivot(index="step", columns="label", values="fraction").fillna(0.0)
    y_black = piv.get(BLACK, pd.Series(0, index=piv.index))
    y_white = piv.get(WHITE, pd.Series(0, index=piv.index))
    y_other = 1.0 - (y_black + y_white)
    xs = piv.index.values
    plt.figure(figsize=(10, 5))
    plt.stackplot(xs,
                  y_black.values, y_white.values, y_other.values,
                  labels=[BLACK, WHITE, "other"],
                  colors=[colors.get(BLACK, "#000000"),
                          colors.get(WHITE, "#ffffff"),
                          "#cccccc"])
    plt.xlabel("Step"); plt.ylabel("Fraction of grid"); plt.title(title)
    plt.legend(loc="upper right")
    save_plot(out_png)

def plot_black_white_only(df_long, colors, out_png: Path, title="Black & White fractions"):
    piv = df_long.pivot(index="step", columns="label", values="fraction").fillna(0.0)
    xs = piv.index.values
    yb = piv.get(BLACK, pd.Series(0, index=piv.index)).values
    yw = piv.get(WHITE, pd.Series(0, index=piv.index)).values
    plt.figure(figsize=(10, 4))
    plt.plot(xs, yb, label=BLACK, color=colors.get(BLACK, "#000"))
    plt.plot(xs, yw, label=WHITE, color=colors.get(WHITE, "#fff"))
    plt.xlabel("Step"); plt.ylabel("Fraction"); plt.title(title)
    plt.legend()
    save_plot(out_png)

def plot_black_white_by_region(df_reg, colors, out_png: Path, title="Black/White by region"):
    reg_order = ["corner", "edge", "interior"]
    plt.figure(figsize=(11, 5))
    for r, region in enumerate(reg_order, start=1):
        sub = df_reg[(df_reg["region"] == region) & (df_reg["label"].isin([BLACK, WHITE]))]
        piv = sub.pivot(index="step", columns="label", values="fraction").fillna(0.0)
        ax = plt.subplot(1, 3, r)
        if BLACK in piv: ax.plot(piv.index, piv[BLACK], color=colors.get(BLACK, "#000"), label=BLACK)
        if WHITE in piv: ax.plot(piv.index, piv[WHITE], color=colors.get(WHITE, "#fff"), label=WHITE)
        ax.set_title(region); ax.set_xlabel("Step"); ax.set_ylim(0, 1)
        if r == 1: ax.set_ylabel("Fraction"); ax.legend()
    plt.suptitle(title, y=1.02, fontsize=12)
    save_plot(out_png)

def plot_diff_curve(states, out_png: Path, title="Cells changed per step"):
    d = diff_curve(states)
    plt.figure(figsize=(10, 3.5))
    plt.plot(np.arange(len(d)), d, marker='.', linewidth=1)
    plt.xlabel("Step t"); plt.ylabel("# changed cells"); plt.title(title)
    save_plot(out_png)

def plot_front_over_time(states, labels, out_png: Path, title="Estimated black front (ℓ∞)"):
    black_idx = labels.index(BLACK)
    radii = [l_inf_front_radius(S, black_idx) for S in states]
    plt.figure(figsize=(10, 3.5))
    plt.plot(radii, marker='.')
    plt.xlabel("Step"); plt.ylabel("Front distance from edge (cells)"); plt.title(title)
    save_plot(out_png)

def plot_stacked_all_labels(df_long, colors, out_png, title="Fractions (stacked, all labels)"):
    """
    Stacked area plot of all labels individually over time.
    """
    piv = df_long.pivot(index="step", columns="label", values="fraction").fillna(0.0)
    xs = piv.index.values
    labs = piv.columns.tolist()
    ys = [piv[lab].values for lab in labs]

    plt.figure(figsize=(10,6))
    plt.stackplot(xs, *ys,
                  labels=labs,
                  colors=[colors.get(lab, "#cccccc") for lab in labs])
    plt.xlabel("Step")
    plt.ylabel("Fraction of grid")
    plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1,0.5))
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_png}")

# ----------------- batch runner used by the CLI -----------------

def run_fractions_reports(run_dirs, outdir: Path, colors_path="label_colors.json", smooth=9):
    """
    Generate CSVs and plots for each run directory.
    Called by the CLI subcommand: bbx fractions --runs ...
    """
    colors = load_label_colors(colors_path)
    outdir.mkdir(parents=True, exist_ok=True)

    for rd in run_dirs:
        rd = Path(rd)
        states, labels, _ = load_run_encoded(rd)

        # Fractions (raw + smoothed)
        df = compute_color_fractions_df(states, labels)
        df_sm = smoothed_per_label(df, window=smooth)

        # Write CSVs
        df_path = outdir / f"{rd.name}_fractions.csv"
        df_sm_path = outdir / f"{rd.name}_fractions_smoothed_w{smooth}.csv"
        df.to_csv(df_path, index=False); print(f"[saved] {df_path}")
        df_sm.to_csv(df_sm_path, index=False); print(f"[saved] {df_sm_path}")

        # Plots → PNGs
        plot_color_fractions(
            df, colors,
            out_png=outdir / f"{rd.name}_fractions_all.png",
            title=f"Color fractions over time — {rd.name}"
        )
        plot_stacked_black_white_other(
            df_sm, colors,
            out_png=outdir / f"{rd.name}_fractions_stacked_bwo.png",
            title=f"Black / White / Other (smoothed w={smooth}) — {rd.name}"
        )
        plot_black_white_only(
            df_sm, colors,
            out_png=outdir / f"{rd.name}_fractions_bw.png",
            title=f"Black & White (smoothed w={smooth}) — {rd.name}"
        )
        df_reg = fractions_by_region(states, labels)
        plot_black_white_by_region(
            df_reg, colors,
            out_png=outdir / f"{rd.name}_bw_by_region.png",
            title=f"Black/White by region — {rd.name}"
        )
        plot_diff_curve(
            states,
            out_png=outdir / f"{rd.name}_diff_curve.png",
            title=f"Cells changed per step — {rd.name}"
        )
        plot_front_over_time(
            states, labels,
            out_png=outdir / f"{rd.name}_front_radius.png",
            title=f"Estimated black front (ℓ∞) — {rd.name}"
        )
        out_stacked_all = outdir / f"{rd.name}_fractions_stacked_all.png"
        plot_stacked_all_labels(df, colors, out_png=out_stacked_all,
                            title=f"Fractions (all labels) — {rd.name}")

# ----------------- CLI / main -----------------

def main():
    ap = argparse.ArgumentParser(description="Report color fractions and evolution plots.")
    ap.add_argument("--run", required=True, help="Run directory, e.g., data/run_000")
    ap.add_argument("--out", default="reports", help="Output reports directory")
    ap.add_argument("--colors", default="label_colors.json", help="Path to label_colors.json")
    ap.add_argument("--smooth", type=int, default=9, help="Rolling window for smoothing (odd recommended)")
    args = ap.parse_args()

    run_dir = Path(args.run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    states, labels, _ = load_run_encoded(run_dir)
    colors = load_label_colors(args.colors)

    # Fractions (raw + smoothed)
    df = compute_color_fractions_df(states, labels)
    df_sm = smoothed_per_label(df, window=args.smooth)

    # Write CSVs
    df_path = out_dir / f"{run_dir.name}_fractions.csv"
    df_sm_path = out_dir / f"{run_dir.name}_fractions_smoothed_w{args.smooth}.csv"
    df.to_csv(df_path, index=False); print(f"[saved] {df_path}")
    df_sm.to_csv(df_sm_path, index=False); print(f"[saved] {df_sm_path}")

    # Plots → PNGs in reports/
    plot_color_fractions(
        df, colors,
        out_png=out_dir / f"{run_dir.name}_fractions_all.png",
        title=f"Color fractions over time — {run_dir.name}"
    )
    plot_stacked_black_white_other(
        df_sm, colors,
        out_png=out_dir / f"{run_dir.name}_fractions_stacked_bwo.png",
        title=f"Black / White / Other (smoothed w={args.smooth}) — {run_dir.name}"
    )
    plot_black_white_only(
        df_sm, colors,
        out_png=out_dir / f"{run_dir.name}_fractions_bw.png",
        title=f"Black & White (smoothed w={args.smooth}) — {run_dir.name}"
    )
    df_reg = fractions_by_region(states, labels)
    plot_black_white_by_region(
        df_reg, colors,
        out_png=out_dir / f"{run_dir.name}_bw_by_region.png",
        title=f"Black/White by region — {run_dir.name}"
    )
    plot_diff_curve(
        states,
        out_png=out_dir / f"{run_dir.name}_diff_curve.png",
        title=f"Cells changed per step — {run_dir.name}"
    )
    plot_front_over_time(
        states, labels,
        out_png=out_dir / f"{run_dir.name}_front_radius.png",
        title=f"Estimated black front (ℓ∞) — {run_dir.name}"
    )
    out_stacked_all = outdir / f"{run_name}_fractions_stacked_all.png"
    plot_stacked_all_labels(df, colors, out_png=out_stacked_all,
                        title=f"Fractions (all labels) — {run_name}")

if __name__ == "__main__":
    main()
