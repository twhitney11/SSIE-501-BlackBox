#!/usr/bin/env python3
"""
bbx: one-stop CLI for Tyler's black box CA.

Subcommands:
  collect      - scrape the PHP page and write runs
  process      - summaries, coverage, conflicts
  analyze      - period scan, classifier, simulate
  fractions    - black/white/other plots + CSVs
  colors       - validate/show label_colors.json
"""

import argparse
from pathlib import Path

# Import your modules (these must exist as you have them now)
from . import process as proc
from . import analyze as an
from . import fractions as fr
from .viz import load_label_colors

# ---------- helpers ----------

def path(p): return Path(p)

def add_common_out(ap):
    ap.add_argument("--out", default="reports", help="Output folder (default: reports)")

# ---------- subcommand runners ----------

def cmd_collect(args):
    from .collect import run_collect  # lazy import
    run_collect(
        base_url=args.base_url,
        steps=args.steps,
        outdir=path(args.out),
        run_prefix=args.run_prefix,
        sleep_ms=args.sleep_ms,
        reset=not args.no_reset,
        seed=args.seed,
    )

def cmd_process(args):
    # Basic summaries (what you already print)
    run_dirs = [path(p) for p in args.runs]
    # Reuse your existing process entrypoints as needed:
    for rd in run_dirs:
        enc_states, labels, _ = proc.load_run_encoded(rd)
        # Per-run summary
        uniq_count, _ = proc.coverage(enc_states, r=args.radius)
        print(f"=== {rd} ===")
        print(f"Steps: {len(enc_states)} | Grid size: {enc_states[0].shape}")
        print(f"Labels: {len(labels)} -> {labels}")
        print(f"Unique neighborhoods (r={args.radius}): {uniq_count}")

def cmd_analyze(args):
    outdir = path(args.out)
    an.run_analysis(
        run_dirs=args.runs,
        outdir=outdir,
        radius=args.radius,
        tests=args.tests,
        period_scan_mode=args.period_scan,
        train_clf=args.train_clf,
        clf_period=args.clf_period,
        do_cycle=args.do_cycle,
        do_ring=args.do_ring,
        do_torus=args.do_torus,
        do_autocorr=args.autocorr,
        do_near_cycle=args.near_cycle,
        near_cycle_maxlag=args.near_cycle_maxlag,
        near_cycle_eps=args.near_cycle_eps,
        perturb_spec=args.perturb,
        simulate_spec=args.simulate,
        sim_steps=args.sim_steps,
        sim_save_every=args.sim_save_every,
        sim_png=args.sim_png,
        sim_seed_colors=args.sim_colors,
        build_rulebook=args.build_rulebook,
        rb_k=args.rb_k,
        rb_split=args.rb_split,
        rb_name=args.rb_name,
        simulate_rulebook_from=args.simulate_rulebook,
        rb_steps=args.rb_steps,
        rb_fallback=args.rb_fallback,
        rb_png=args.rb_png,
        rb_save_every=args.rb_save_every,
        rb_colors=args.rb_colors
    )


def cmd_fractions(args):
    outdir = path(args.out)
    fr.run_fractions_reports(
        run_dirs=args.runs,
        outdir=outdir,
        colors_path=args.colors,
        smooth=args.smooth
    )

def cmd_colors(args):
    colors = load_label_colors(args.colors)
    print("Loaded label → hex map:")
    for k, v in colors.items():
        print(f"  {k:>4} : {v}")

def cmd_rbxplore(args):
    from . import rbxplore as rb  # lazy import
    outdir = path(args.out)
    rbk = rb.load_rulebooks(args.rulebook)

    if args.all or args.stats:
        rb.rb_stats(rbk, outdir)
    if args.all or args.symmetry:
        rb.symmetry_checks(rbk, outdir=outdir)
    if args.all or args.phase_delta:
        rb.phase_delta(rbk, outdir)
    if args.all or args.collapse:
        rb.three_by_three_collapse(rbk, outdir)
    if args.all or args.saliency:
        rb.position_saliency(rbk, outdir)
    if args.all or args.mutual_info:
        rb.mutual_info_positions(rbk, outdir)
    if args.all or args.tree:
        rb.train_tree_surrogate(rbk, outdir, max_depth=args.tree_depth)

def cmd_gridmaps(args):
    from . import gridmaps as gm
    gm.run_gridmaps(args.runs, args.out)

# ---------- main CLI ----------

def main():
    ap = argparse.ArgumentParser(prog="bbx", description="Black Box CA toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # collect
    apc = sub.add_parser("collect", help="Scrape the PHP page and record a run")
    apc.add_argument("--base-url", required=True, help="URL of BlackBox.php")
    apc.add_argument("--steps", type=int, default=500)
    apc.add_argument("--out", default="data")
    apc.add_argument("--run-prefix", default="run")
    apc.add_argument("--sleep-ms", type=int, default=0)
    apc.add_argument("--seed", type=int, default=None, help="Optional seed value")
    apc.add_argument("--no-reset", action="store_true", help="Do not send ?reset=1 before capture")
    apc.set_defaults(func=cmd_collect)

    # process
    app = sub.add_parser("process", help="Summarize runs (coverage, labels)")
    app.add_argument("--runs", nargs="+", required=True)
    app.add_argument("--radius", type=int, default=1)
    app.set_defaults(func=cmd_process)

    # analyze
    apa = sub.add_parser("analyze", help="Period scan, classifier, simulate")
    apa.add_argument("--runs", nargs="+", required=True)
    apa.add_argument("--radius", type=int, default=1)
    apa.add_argument("--tests", default="", help="shape,conflicts,permutation,totalistic")
    apa.add_argument("--period-scan", default="", help="none,edge,corner_edge (comma-sep)")
    apa.add_argument("--train-clf", action="store_true")
    apa.add_argument("--clf-period", default="")
    apa.add_argument("--do-cycle", action="store_true", help="Disable cycle detection")
    apa.add_argument("--do-ring", action="store_true", help="Disable ring tracking")
    apa.add_argument("--do-torus", action="store_true", help="Disable torus wrap test")
    apa.add_argument("--autocorr", action="store_true", help="Autocorrelation of black/white fractions and ring radius (run_0)")
    apa.add_argument("--near-cycle", action="store_true", help="Near-cycle scan on B/W/Other-collapsed states (run_0)")
    apa.add_argument("--near-cycle-maxlag", type=int, default=64, help="Max lag for near-cycle scan (default 64)")
    apa.add_argument("--near-cycle-eps", type=float, default=0.01, help="Mismatch fraction threshold for a near-cycle hit (default 0.01)")
    apa.add_argument("--perturb", default="", help="Perturbation spec, e.g. 'block:cx=10,cy=10,w=10,h=10,mode=rand,steps=150,base_step=-100'")
    apa.add_argument("--simulate", default="", help="Simulate with the exported classifier from a seed. Examples: 'run:idx=0', 'run:last', 'file:data/run_000/step_0050.json'")
    apa.add_argument("--sim-steps", type=int, default=0, help="Number of steps to roll forward (required if --simulate)")
    apa.add_argument("--sim-save-every", type=int, default=0, help="If >0, save decoded JSON grid every k steps plus final")
    apa.add_argument("--sim-png", action="store_true", help="Also save a PNG panel of snapshots")
    apa.add_argument("--sim-colors", default="label_colors.json", help="Path to label_colors.json for PNG coloring")
    apa.add_argument("--build-rulebook", action="store_true", help="Build phase rulebooks from run_0 and save to reports/<rb_name>.json")
    apa.add_argument("--rb-k", default="", help="Phase period k (defaults to 2 if omitted)")
    apa.add_argument("--rb-split", default="none", choices=["none","edge","corner_edge"], help="Region split for rulebooks")
    apa.add_argument("--rb-name", default="rulebooks", help="Base filename for rulebook JSON in reports/")
    apa.add_argument("--simulate-rulebook", default="", help="Seed for rulebook sim: 'run:idx=0' | 'run:last' | 'file:path.json'")
    apa.add_argument("--rb-steps", type=int, default=0, help="Steps to roll with rulebook (required if --simulate-rulebook)")
    apa.add_argument("--rb-fallback", default="copy", choices=["copy","error"], help="On unseen neighborhood: 'copy' (stay same) or 'error'")
    apa.add_argument("--rb-png", action="store_true", help="Save PNG montage for rulebook sim")
    apa.add_argument("--rb-save-every", type=int, default=0, help="Save decoded JSON every k steps")
    apa.add_argument("--rb-colors", default="label_colors.json", help="Colors for PNG montage")

    add_common_out(apa)
    apa.set_defaults(func=cmd_analyze)

    # fractions
    apf = sub.add_parser("fractions", help="Fractions + region + diff/front plots")
    apf.add_argument("--runs", nargs="+", required=True)
    apf.add_argument("--colors", default="label_colors.json")
    apf.add_argument("--smooth", type=int, default=9)
    add_common_out(apf)
    apf.set_defaults(func=cmd_fractions)

    # colors
    apl = sub.add_parser("colors", help="Print the color legend")
    apl.add_argument("--colors", default="label_colors.json")
    apl.set_defaults(func=cmd_colors)

    # rbxplore
    apr = sub.add_parser("rbxplore", help="Explore rulebooks (stats, symmetry, saliency, etc.)")
    apr.add_argument("--rulebook", default="reports/rulebooks.json", help="Path to rulebooks.json")
    apr.add_argument("--out", default="reports", help="Output directory")
    apr.add_argument("--all", action="store_true", help="Run all analyses")
    apr.add_argument("--stats", action="store_true", help="Output per-phase/region output distributions")
    apr.add_argument("--symmetry", action="store_true", help="Check invariance under flips/rotations")
    apr.add_argument("--phase-delta", action="store_true", help="Compare outputs across phases")
    apr.add_argument("--collapse", action="store_true", help="Test 5x5→3x3 collapse conflicts")
    apr.add_argument("--saliency", action="store_true", help="Mask positions and check ambiguity")
    apr.add_argument("--mutual-info", action="store_true", help="Mutual information per position")
    apr.add_argument("--tree", action="store_true", help="Train a decision tree surrogate")
    apr.add_argument("--tree-depth", type=int, default=6, help="Max depth for surrogate tree")
    apr.set_defaults(func=cmd_rbxplore)

    # heatmaps
    apg = sub.add_parser("gridmaps", help="Make full 20×20 spatial heatmaps")
    apg.add_argument("--runs", nargs="+", required=True)
    apg.add_argument("--out", default="reports")
    apg.set_defaults(func=cmd_gridmaps)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
