import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

DEFAULT_SPLITS = ("none", "edge", "corner_edge")
REQUIRED_COLUMNS = {"k", "split", "phase", "region", "conflicts", "rule_size"}


def _load_period_scan_frames(outdir: Path, splits: Sequence[str]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for split in splits:
        path = outdir / f"period_scan_{split}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]  # normalise headers
        frames.append(df)
    return frames


def _top_zero_ks(group: pd.DataFrame, limit: int = 10) -> list[int]:
    zero_rows = group[group["conflicts"] == 0].sort_values(["k", "phase"])
    ks = zero_rows["k"].drop_duplicates().head(limit)
    return [int(k) for k in ks]


def _summarise_frames(frames: Iterable[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.concat(frames, ignore_index=True)
    missing = REQUIRED_COLUMNS - set(full.columns)
    if missing:
        raise ValueError(f"Missing columns in period scan CSVs: {sorted(missing)}")

    rows = []
    zero_lists = []
    for region, group in full.groupby("region"):
        group = group.sort_values(["k", "conflicts", "phase"])
        zeros = group[group["conflicts"] == 0]
        record = {
            "region": region,
            "first_zero_k": None,
            "phases_at_first_zero": [],
            "num_zero_rows": 0,
            "best_k": int(group.iloc[0]["k"]),
            "best_conflicts": int(group.iloc[0]["conflicts"]),
        }
        if not zeros.empty:
            first_zero_k = int(zeros["k"].min())
            phases = sorted(zeros.loc[zeros["k"] == first_zero_k, "phase"].unique().tolist())
            record.update({
                "first_zero_k": first_zero_k,
                "phases_at_first_zero": phases,
                "num_zero_rows": int(len(zeros)),
                "best_k": first_zero_k,
                "best_conflicts": 0,
            })
        rows.append(record)
        zero_lists.append({"region": region, "zero_k_values": _top_zero_ks(group)})

    first_zero = pd.DataFrame(rows).sort_values("region").reset_index(drop=True)
    zero_values = pd.DataFrame(zero_lists).sort_values("region").reset_index(drop=True)
    return first_zero, zero_values


def summarise_period_scan(outdir: Path, splits: Sequence[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Combine period_scan_*.csv files and emit summary CSVs highlighting low-conflict k values.
    Returns (first_zero_df, zero_values_df) or None if no inputs found.
    """
    outdir = Path(outdir)
    splits = tuple(splits) if splits else DEFAULT_SPLITS
    frames = _load_period_scan_frames(outdir, splits)
    if not frames:
        print(f"[best_k] No period_scan CSVs found in {outdir} for splits: {', '.join(splits)}")
        return None

    first_zero, zero_lists = _summarise_frames(frames)
    first_zero_path = outdir / "period_first_zero_by_region.csv"
    zero_list_path = outdir / "period_zero_k_list_by_region.csv"
    first_zero.to_csv(first_zero_path, index=False)
    zero_lists.to_csv(zero_list_path, index=False)
    print(f"[best_k] Saved {first_zero_path}")
    print(f"[best_k] Saved {zero_list_path}")
    return first_zero, zero_lists


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise period_scan CSVs to highlight conflict-free k values.")
    ap.add_argument("--out", default="reports", type=Path, help="Directory containing period_scan_*.csv files")
    ap.add_argument("--splits", default="none,edge,corner_edge",
                    help="Comma-separated list of split modes to include (default: none,edge,corner_edge)")
    args = ap.parse_args()
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    summarise_period_scan(args.out, splits)


if __name__ == "__main__":
    main()
