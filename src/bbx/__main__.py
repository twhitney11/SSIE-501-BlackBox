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
"""

import argparse
from pathlib import Path

from . import process as proc
from . import analyze as an
from . import fractions as fr
from .viz import load_label_colors


# ---------- helpers ----------
def _p(p): return Path(p)
def _add_out(ap): ap.add_argument("--out", default="reports", help="Output folder (default: reports)")


# ---------- subcommand runners ----------
def cmd_collect(args):
    # collect.py must expose run_collect(base_url, steps, outdir, run_prefix, sleep_ms, reset, seed)
    from .collect import run_collect  # lazy import
    run_collect(
        base_url=args.base_url,
        steps=args.steps,
        outdir=_p(args.out),
        run_prefix=args.run_prefix,
        sleep_ms=args.sleep_ms,
        reset=not args.no_reset,
        seed=args.seed,
    )


def cmd_process(args):
    run_dirs = [_p(p) for p in args.runs]
    for rd in run_dirs:
        enc_states, labels, _ = proc.load_run_encoded(rd)
        uniq_count, _ = proc.coverage(enc_states, r=args.radius)
        print(f"=== {rd} ===")
        print(f"Steps: {len(enc_states)} | Grid size: {enc_states[0].shape}")
        print(f"Labels: {len(labels)} -> {labels}")
        print(f"Unique neighborhoods (r={args.radius}): {uniq_count}")


def cmd_analyze(args):
    outdir = _p(args.out)
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
    )


def cmd_fractions(args):
    outdir = _p(args.out)
    fr.run_fractions_reports(
        run_dirs=args.runs,
        outdir=outdir,
        colors_path=args.colors,
        smooth=args.smooth,
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
    apc.add_argument("--no-reset", action="store_true", help="Skip sending ?reset=1 before capture")
    apc.set_defaults(func=cmd_collect)

    # process
    app = sub.add_parser("process", help="Summarize runs (coverage, labels)")
    app.add_argument("--runs", nargs="+", required=True)
    app.add_argument("--radius", type=int, default=1)
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

    _add_out(apa)
    apa.set_defaults(func=cmd_analyze)

    # fractions
    apf = sub.add_parser("fractions", help="Fractions + region + diff/front plots")
    apf.add_argument("--runs", nargs="+", required=True)
    apf.add_argument("--colors", default="label_colors.json")
    apf.add_argument("--smooth", type=int, default=9)
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

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
