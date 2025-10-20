#!/usr/bin/env python3
"""
bbx: one-stop CLI for black box CA.

Subcommands:
  collect      - scrape the PHP page and write runs
  process      - summaries, coverage, conflicts
  analyze      - period scan, classifier, simulate, rulebooks, diagnostics
  fractions    - black/white/other plots + CSVs
  colors       - validate/show label_colors.json
  rbxplore     - explore/simplify rulebooks (symmetry, MI, small tree)
  gridmaps     - full 20×20 spatial heatmaps (change rate, entropy, etc.)
  regionize    - derive wall/inside/outside masks & stats
  animate      - render animated GIFs of runs/windows
  gof          - goodness-of-fit diagnostics for local models
  aggregate    - cross-run mean/CI summaries
  entropy      - neighborhood & conditional entropy analysis
  mi           - mutual information & correlation vs distance
  stationarity - stationarity / cross-run generalization
"""

import argparse
from pathlib import Path

import numpy as np

from . import process as proc
from . import analyze as an
from . import fractions as fr
from .viz import load_label_colors
from .maskgen import generate_masks
# ---------- helpers ----------
def _p(p): return Path(p)
def _add_out(ap): ap.add_argument("--out", default="reports", help="Output folder (default: reports)")


# ---------- subcommand runners ----------
def cmd_collect(args):
    from .collect import run_collect, run_collect_batch  # lazy import
    outdir = _p(args.out)
    reset = not args.no_reset
    if args.runs and args.runs > 1:
        run_collect_batch(
            base_url=args.base_url,
            steps=args.steps,
            outdir=outdir,
            runs=args.runs,
            run_prefix=args.run_prefix,
            sleep_ms=args.sleep_ms,
            reset=reset,
            seed=args.seed,
        )
    else:
        run_collect(
            base_url=args.base_url,
            steps=args.steps,
            outdir=outdir,
            run_prefix=args.run_prefix,
            sleep_ms=args.sleep_ms,
            reset=reset,
            seed=args.seed,
        )


def cmd_process(args):
    run_dirs = [_p(p) for p in args.runs]
    for rd in run_dirs:
        proc.summarize_run(rd, radius=args.radius, window=args.window)
    if len(run_dirs) > 1:
        proc.summarize_runs(run_dirs, radius=args.radius, window=args.window)


def cmd_analyze(args):
    outdir = _p(args.out)
    scope_mask = None
    scope_desc = None
    if args.scope_mask and args.scope_region:
        raise ValueError("Provide at most one of --scope-mask or --scope-region.")
    if args.scope_mask:
        if ":" in args.scope_mask:
            mask_path_str, column = args.scope_mask.split(":", 1)
        else:
            mask_path_str, column = args.scope_mask, None
        mask_path = Path(mask_path_str)
        scope_mask = an.load_scope_mask_csv(mask_path, column=column)
        scope_desc = f"mask:{mask_path.name}" + (f":{column}" if column else "")
    elif args.scope_region:
        if ":" not in args.scope_region:
            raise ValueError("--scope-region must be provided as 'path:region_name'.")
        mask_path_str, region_name = args.scope_region.rsplit(":", 1)
        mask_path = Path(mask_path_str)
        scope_mask = an.load_scope_region_mask(mask_path, region_name)
        scope_desc = f"region:{region_name}:{mask_path.name}"
    if scope_mask is not None and args.scope_invert:
        scope_mask = np.logical_not(scope_mask)
        scope_desc = (scope_desc or "mask") + " (inverted)"

    an.run_analysis(
        run_dirs=args.runs,
        outdir=outdir,
        radius=args.radius,
        tests=args.tests,
        period_scan_mode=args.period_scan,
        train_clf=args.train_clf,
        clf_period=args.clf_period,

        # cycle/shape diagnostics
        do_cycle=args.do_cycle,
        do_ring=args.do_ring,
        do_torus=args.do_torus,
        do_autocorr=args.autocorr,
        do_near_cycle=args.near_cycle,
        near_cycle_maxlag=args.near_cycle_maxlag,
        near_cycle_eps=args.near_cycle_eps,

        # perturb & simulate (classifier)
        perturb_spec=args.perturb,
        simulate_spec=args.simulate,
        sim_steps=args.sim_steps,
        sim_save_every=args.sim_save_every,
        sim_png=args.sim_png,
        sim_seed_colors=args.sim_colors,

        # rulebook build & simulate
        build_rulebook=args.build_rulebook,
        rb_k=args.rb_k,
        rb_split=args.rb_split,
        rb_name=args.rb_name,
        simulate_rulebook_from=args.simulate_rulebook,
        rb_steps=args.rb_steps,
        rb_fallback=args.rb_fallback,
        rb_png=args.rb_png,
        rb_save_every=args.rb_save_every,
        rb_colors=args.rb_colors,
        window=args.window,
        scope_mask=scope_mask,
        scope_desc=scope_desc,
    )


def cmd_masks(args):
    config_path = _p(args.config)
    outdir = _p(args.out)
    run = args.run if args.run else None
    generate_masks(config_path, outdir, run=run, window=args.window, allow_overlap=args.allow_overlap)


def cmd_fractions(args):
    outdir = _p(args.out)
    region_labels = [lab.strip() for lab in args.region_labels.split(",") if lab.strip()]
    fr.run_fractions_reports(
        run_dirs=args.runs,
        outdir=outdir,
        colors_path=args.colors,
        smooth=args.smooth,
        region_labels=region_labels,
        window=args.window,
    )


def cmd_colors(args):
    colors = load_label_colors(args.colors)
    print("Loaded label → hex map:")
    for k, v in colors.items():
        print(f"  {k:>4} : {v}")


def cmd_rbxplore(args):
    # rbxplore.py must expose rb_load + rb_* functions we added
    from . import rbxplore as rb
    outdir = _p(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = rb.rb_load(args.rb)
    todo = set(args.do.split(",")) if args.do else {"all"}

    if "all" in todo or "orient" in todo:
        rb.rb_orientation_invariance(meta, outdir)
    if "all" in todo or "center" in todo:
        rb.rb_center_dependence(meta, outdir)
    if "all" in todo or "mi" in todo:
        rb.rb_positional_mi(meta, outdir)
    if "all" in todo or "tree" in todo:
        rb.rb_small_tree(
            meta, outdir,
            phase=args.tree_phase,
            region=args.tree_region,
            max_depth=args.tree_depth,
        )

def cmd_gridmaps(args):
    from . import gridmaps as gm
    gm.run_gridmaps(
        args.runs,
        args.out,
        win=args.win,
        phase=args.phase,
        k=args.k,
        sequence=args.sequence,
        sequence_steps=args.sequence_steps,
        sequence_file=args.sequence_file,
        colors_path=args.colors,
        sequence_show=args.sequence_show,
    )

def cmd_regionize(args):
    from .regionize import RegionizeConfig, run_regionize
    cfg = RegionizeConfig(
        run_dir=Path(args.run),
        outdir=Path(args.out),
        window=args.window,
        label=args.label,
        threshold=args.threshold,
        wall_thickness=args.wall_thickness,
        flip_period=args.flip_period,
    )
    run_regionize(cfg)

def cmd_animate(args):
    from .animate import generate_gif
    if args.fps is not None:
        if args.fps <= 0:
            raise ValueError("--fps must be positive")
        duration_ms = max(1, int(round(1000.0 / args.fps)))
    else:
        duration_ms = args.duration if args.duration is not None else 100
        if duration_ms <= 0:
            raise ValueError("--duration must be positive")
    stride = max(1, args.stride)
    scale = max(1, args.scale)
    loop = max(0, args.loop)
    generate_gif(
        run_dir=Path(args.run),
        out_path=Path(args.out),
        window=args.window,
        stride=stride,
        scale=scale,
        duration_ms=duration_ms,
        loop=loop,
        colors_path=Path(args.colors),
    )


def cmd_gof(args):
    from .gof import GOFConfig, run_gof

    perms = [p.strip() for p in (args.permutations or "").split(",") if p.strip()]
    cfg = GOFConfig(
        train_run=_p(args.train_run),
        train_window=args.train_window or None,
        test_runs=[_p(p) for p in (args.test_runs if args.test_runs else [args.train_run])],
        test_window=args.test_window or None,
        radius=args.radius,
        feature_mode=args.feature_mode,
        model=args.model,
        max_samples=args.max_samples if args.max_samples else None,
        permutations=perms,
        seed=args.seed,
        outdir=_p(args.out),
        knn_k=args.knn_k,
    )
    run_gof(cfg)


def cmd_aggregate(args):
    from .aggregate import run_aggregate
    run_aggregate(args)

def cmd_entropy(args):
    from .entropy import run_entropy
    run_entropy(args)

def cmd_mi(args):
    from .mi import run_mi
    run_mi(args)

def cmd_stationarity(args):
    from .stationarity import run_stationarity
    run_stationarity(args)
# ---------- main CLI ----------
def main():
    ap = argparse.ArgumentParser(prog="bbx", description="Black Box CA toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # collect
    apc = sub.add_parser("collect", help="Scrape the PHP page and record a run")
    apc.add_argument("--base-url", required=True, help="URL of BlackBox.php")
    apc.add_argument("--steps", type=int, default=500)
    apc.add_argument("--out", default="data", help="Output folder (default: data)")
    apc.add_argument("--run-prefix", default="run", help="Subfolder prefix (default: run)")
    apc.add_argument("--sleep-ms", type=int, default=0, help="Delay between steps (ms)")
    apc.add_argument("--seed", type=int, default=None, help="Optional seed value")
    apc.add_argument("--runs", type=int, default=1, help="Number of independent runs to capture (default: 1)")
    apc.add_argument("--no-reset", action="store_true", help="Skip sending ?reset=1 before capture")
    apc.set_defaults(func=cmd_collect)

    # process
    app = sub.add_parser("process", help="Summarize runs (coverage, labels)")
    app.add_argument("--runs", nargs="+", required=True)
    app.add_argument("--radius", type=int, default=1)
    app.add_argument("--window", default="", help="time window like 'start:end' or 'last:N'")
    app.set_defaults(func=cmd_process)

    # analyze
    apa = sub.add_parser("analyze", help="Period scan, classifier, simulate, rulebooks")
    apa.add_argument("--runs", nargs="+", required=True)
    apa.add_argument("--radius", type=int, default=1)
    apa.add_argument("--tests", default="", help="Comma-set: shape,conflicts,permutation,totalistic (empty=all)")
    apa.add_argument("--period-scan", default="", help="Comma-set of: none,edge,corner_edge (empty=skip)")
    apa.add_argument("--train-clf", action="store_true", help="Train classifier (multinomial LR)")
    apa.add_argument("--clf-period", default="", help="Override time period k for classifier (int)")

    # diagnostics & dynamics
    apa.add_argument("--do-cycle", action="store_true", help="Run cycle detection diagnostics")
    apa.add_argument("--do-ring", action="store_true", help="Track black-ring radius over time")
    apa.add_argument("--do-torus", action="store_true", help="Wrap edges (torus) experiment")
    apa.add_argument("--autocorr", action="store_true", help="Autocorrelation of B/W fractions & ring radius (run_0)")
    apa.add_argument("--near-cycle", action="store_true", help="Near-cycle scan on collapsed states (run_0)")
    apa.add_argument("--near-cycle-maxlag", type=int, default=64)
    apa.add_argument("--near-cycle-eps", type=float, default=0.01)

    # perturbations / simulate with classifier
    apa.add_argument("--perturb", default="", help="e.g. 'block:cx=10,cy=10,w=10,h=10,mode=rand,steps=150,base_step=-100'")
    apa.add_argument("--simulate", default="", help="Seed for classifier sim: 'run:idx=0' | 'run:last' | 'file:...json'")
    apa.add_argument("--sim-steps", type=int, default=0)
    apa.add_argument("--sim-save-every", type=int, default=0)
    apa.add_argument("--sim-png", action="store_true")
    apa.add_argument("--sim-colors", default="label_colors.json")

    # rulebook build & simulate
    apa.add_argument("--build-rulebook", action="store_true", help="Build phase rulebooks from run_0")
    apa.add_argument("--rb-k", default="", help="Phase period k (default 2 if omitted)")
    apa.add_argument("--rb-split", default="none", choices=["none","edge","corner_edge"])
    apa.add_argument("--rb-name", default="rulebooks")
    apa.add_argument("--simulate-rulebook", default="", help="Seed for rulebook sim: 'run:idx=0' | 'run:last' | 'file:...json'")
    apa.add_argument("--rb-steps", type=int, default=0)
    apa.add_argument("--rb-fallback", default="copy", choices=["copy","error"])
    apa.add_argument("--rb-png", action="store_true")
    apa.add_argument("--rb-save-every", type=int, default=0)
    apa.add_argument("--rb-colors", default="label_colors.json")
    apa.add_argument("--window", default="", help="time window like 'start:end' or 'last:N'")
    apa.add_argument("--scope-mask", default="", help="Mask CSV to restrict analysis cells (path[:column])")
    apa.add_argument("--scope-region", default="", help="Region labels CSV and name (path:region_name)")
    apa.add_argument("--scope-invert", action="store_true", help="Invert the loaded scope mask")

    _add_out(apa)
    apa.set_defaults(func=cmd_analyze)

    # masks
    apm = sub.add_parser("masks", help="Generate manual region masks from config")
    apm.add_argument("--config", required=True, help="Path to mask config JSON")
    apm.add_argument("--out", default="reports/masks", help="Output directory (default: reports/masks)")
    apm.add_argument("--run", default="", help="Optional run directory to infer grid size")
    apm.add_argument("--window", default="", help="Window 'start:end' or 'last:N' when inferring from run")
    apm.add_argument("--allow-overlap", dest="allow_overlap", action="store_true", help="Allow overlapping regions (first region wins where overlap)")
    apm.add_argument("--no-allow-overlap", dest="allow_overlap", action="store_false", help="Disallow overlapping regions (default uses config setting)")
    apm.set_defaults(func=cmd_masks, allow_overlap=None)

    # fractions
    apf = sub.add_parser("fractions", help="Fractions + region + diff/front plots")
    apf.add_argument("--runs", nargs="+", required=True)
    apf.add_argument("--colors", default="label_colors.json")
    apf.add_argument("--smooth", type=int, default=9)
    apf.add_argument("--region-labels", default="gru,mex", help="Comma list of labels for region plot (default: gru,mex)")
    apf.add_argument("--window", default="", help="Time window 'start:end' or 'last:N' (default full run)")
    _add_out(apf)
    apf.set_defaults(func=cmd_fractions)

    # colors
    apl = sub.add_parser("colors", help="Print the color legend")
    apl.add_argument("--colors", default="label_colors.json")
    apl.set_defaults(func=cmd_colors)

    # rbxplore (rulebook exploration/simplification)
    apr = sub.add_parser("rbxplore", help="Explore rulebooks (symmetry, center, MI, small tree)")
    apr.add_argument("--rb", required=True, help="Path to rulebooks.json")
    apr.add_argument("--do", default="all", help="Comma list: orient,center,mi,tree,all")
    apr.add_argument("--tree-phase", type=int, default=0)
    apr.add_argument("--tree-region", default="interior")
    apr.add_argument("--tree-depth", type=int, default=3)
    _add_out(apr)
    apr.set_defaults(func=cmd_rbxplore)

    # gridmaps
    apg = sub.add_parser("gridmaps", help="Make full 20×20 spatial heatmaps")
    apg.add_argument("--runs", nargs="+", required=True)
    apg.add_argument("--out", default="reports")
    apg.add_argument("--win", default="", help="time window like 'start:end' or 'last:N'")
    apg.add_argument("--phase", type=int, default=-1, help="-1=all, 0..k-1 for phase slice")
    apg.add_argument("--k", type=int, default=2, help="phase period (default 2)")
    apg.add_argument("--sequence", action="store_true", help="Also render a step montage from the first run")
    apg.add_argument("--sequence-steps", default="", help="Comma list or 'auto[:N]' for montage sampling")
    apg.add_argument("--sequence-file", default="", help="Optional filename for the montage PNG")
    apg.add_argument("--colors", default="label_colors.json", help="Label→color JSON for the montage")
    apg.add_argument("--sequence-show", action="store_true", help="Display the montage interactively")
    apg.set_defaults(func=cmd_gridmaps)

    # regionize
    aprg = sub.add_parser("regionize", help="Extract wall/inside/outside regions from a run")
    aprg.add_argument("--run", required=True, help="Run directory (e.g., data/run_000)")
    aprg.add_argument("--out", default="reports", help="Output directory (default: reports)")
    aprg.add_argument("--window", default="", help="Time window 'start:end' or 'last:N' (default: full run)")
    aprg.add_argument("--label", default="gru", help="Label to treat as wall (default: gru)")
    aprg.add_argument("--threshold", type=float, default=0.5, help="Occupancy threshold for wall classification (default: 0.5)")
    aprg.add_argument("--wall-thickness", type=int, default=2, help="Wall band thickness in cells (default: 2)")
    aprg.add_argument("--flip-period", type=int, default=2, help="Phase period k for flip-rate calculations (default: 2)")
    aprg.set_defaults(func=cmd_regionize)

    # animate
    apa = sub.add_parser("animate", help="Render an animated GIF for a run")
    apa.add_argument("--run", required=True, help="Run directory (e.g., data/run_000)")
    apa.add_argument("--out", required=True, help="Output GIF path")
    apa.add_argument("--window", default="", help="Time window 'start:end' or 'last:N' (default full run)")
    apa.add_argument("--stride", type=int, default=1, help="Use every Nth frame (default 1)")
    apa.add_argument("--scale", type=int, default=4, help="Pixel up-scale factor (default 4)")
    timing = apa.add_mutually_exclusive_group()
    timing.add_argument("--duration", type=int, default=None, help="Frame duration in ms (default 100)")
    timing.add_argument("--fps", type=float, default=None, help="Frames per second (alternative to --duration)")
    apa.add_argument("--loop", type=int, default=0, help="GIF loop count (0=infinite)")
    apa.add_argument("--colors", default="label_colors.json", help="Path to label→color JSON")
    apa.set_defaults(func=cmd_animate)

    # gof
    apg = sub.add_parser("gof", help="Evaluate local goodness of fit and locality")
    apg.add_argument("--train-run", required=True, help="Run used for training the model")
    apg.add_argument("--train-window", default="", help="Training window 'start:end' or 'last:N'")
    apg.add_argument("--test-runs", nargs="*", default=None, help="Optional test runs (default: train run)")
    apg.add_argument("--test-window", default="", help="Test window 'start:end' or 'last:N'")
    apg.add_argument("--radius", type=int, default=1, help="Neighborhood radius")
    apg.add_argument("--model", choices=["logistic", "rule", "markov", "knn"], default="logistic")
    apg.add_argument("--feature-mode", choices=["local", "center", "global"], default="local")
    apg.add_argument("--max-samples", type=int, default=0, help="Optional subsample limit")
    apg.add_argument("--permutations", default="", help="Comma list of permutation tests (e.g. random)")
    apg.add_argument("--seed", type=int, default=None, help="Random seed for subsampling/permutations")
    apg.add_argument("--knn-k", type=int, default=5, help="k for KNN model (model=knn)")
    _add_out(apg)
    apg.set_defaults(func=cmd_gof)

    # aggregate
    apa = sub.add_parser("aggregate", help="Aggregate metrics across runs/windows")
    apa.add_argument("--runs", nargs="+", required=True)
    apa.add_argument("--window", default="", help="Time window 'start:end' or 'last:N'")
    apa.add_argument("--labels", default="", help="Comma list of focus labels")
    apa.add_argument("--colors", default="label_colors.json")
    _add_out(apa)
    apa.set_defaults(func=cmd_aggregate)

    apa = sub.add_parser("entropy", help="Neighborhood / conditional entropy analysis")
    apa.add_argument("--runs", nargs="+", required=True)
    apa.add_argument("--window", default="", help="Time window 'start:end' or 'last:N'")
    apa.add_argument("--radius", type=int, default=1)
    apa.add_argument("--regions", default="all,interior,edge,corner", help="Comma list of regions")
    _add_out(apa)
    apa.set_defaults(func=cmd_entropy)

    apa = sub.add_parser("mi", help="Pairwise mutual information / correlation analysis")
    apa.add_argument("--runs", nargs="+", required=True)
    apa.add_argument("--window", default="", help="Time window 'start:end' or 'last:N'")
    apa.add_argument("--samples", type=int, default=1000, help="Number of sampled cell pairs")
    apa.add_argument("--seed", type=int, default=None)
    _add_out(apa)
    apa.set_defaults(func=cmd_mi)

    apa = sub.add_parser("stationarity", help="Stationarity / cross-run generalization")
    apa.add_argument("--train-run", required=True)
    apa.add_argument("--train-window", default="")
    apa.add_argument("--test-runs", nargs="*", default=None)
    apa.add_argument("--test-window", default="")
    apa.add_argument("--radius", type=int, default=1)
    apa.add_argument("--feature-mode", choices=["local", "center", "global"], default="local")
    apa.add_argument("--model", choices=["logistic", "rule"], default="logistic")
    apa.add_argument("--segments", type=int, default=4)
    apa.add_argument("--max-samples", type=int, default=0)
    apa.add_argument("--seed", type=int, default=None)
    _add_out(apa)
    apa.set_defaults(func=cmd_stationarity)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
