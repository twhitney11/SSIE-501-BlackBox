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
    # Assuming you already have collect.py with `run_collect(...)`
    from .collect import run_collect  # lazy import
    run_collect(
        base_url=args.base_url,
        steps=args.steps,
        seed=args.seed,
        outdir=path(args.out),
        run_prefix=args.run_prefix,
        sleep_ms=args.sleep_ms
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
    # Defer to your analyze.run_analysis with the flags we added
    outdir = path(args.out)
    an.run_analysis(
        run_dirs=args.runs,
        outdir=outdir,
        radius=args.radius,
        tests=args.tests,
        period_scan_mode=args.period_scan,
        train_clf=args.train_clf,
        clf_period=args.clf_period,
        do_cycle=not args.no_cycle,
        do_ring=not args.no_ring,
        do_torus=not args.no_torus,
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

# ---------- main CLI ----------

def main():
    ap = argparse.ArgumentParser(prog="bbx", description="Black Box CA toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # collect
    apc = sub.add_parser("collect", help="Scrape the PHP page and record a run")
    apc.add_argument("--base-url", required=True, help="URL of BlackBox.php page")
    apc.add_argument("--steps", type=int, default=500)
    apc.add_argument("--seed", type=int, default=None, help="Optional seed param")
    apc.add_argument("--run-prefix", default="run", help="Name prefix for output runs")
    apc.add_argument("--sleep-ms", type=int, default=0, help="Delay between requests")
    add_common_out(apc)
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
    apa.add_argument("--no-cycle", action="store_true", help="Disable cycle detection")
    apa.add_argument("--no-ring", action="store_true", help="Disable ring tracking")
    apa.add_argument("--no-torus", action="store_true", help="Disable torus wrap test")
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

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
