# rbxplore.py
from __future__ import annotations
import json, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from sklearn.tree import DecisionTreeClassifier, export_text

# ---------------- IO ----------------

def rb_load(path: str | Path) -> dict:
    """Load rulebooks.json as produced by analyze.py (build-rulebook)."""
    p = Path(path)
    meta = json.loads(p.read_text())
    # Basic shape check
    if "rules" not in meta:
        raise ValueError(f"rulebook missing 'rules': {p}")
    return meta

def _save_csv(rows, path: Path, header=None):
    df = pd.DataFrame(rows, columns=header if header else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[saved] {path}")

# ---------------- helpers ----------------

def _phase_keys_and_regions(rules: dict[str, dict]) -> tuple[list[str], list[str]]:
    """
    Normalize phase keys to str and return (phase_keys_str, region_names).
    Expect: rules[phase][region] = { 'k1,k2,...,k25': out_label }
    If split='none', regions = ['all'].
    """
    phase_keys = sorted([str(k) for k in rules.keys()], key=lambda s: int(s))
    # find a sample region container
    sample = rules[phase_keys[0]]
    if isinstance(sample, dict) and sample and isinstance(next(iter(sample.values())), dict):
        regions = list(sample.keys())
    else:
        regions = ["all"]
    return phase_keys, regions

def _key_to_array(key: str, size=5) -> np.ndarray:
    vals = np.array(list(map(int, key.split(","))), dtype=np.int32)
    return vals.reshape(size, size)

def _array_to_key(arr: np.ndarray) -> str:
    return ",".join(map(str, arr.reshape(-1).tolist()))

def _rotate90(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, k=1)

def _flip_h(arr: np.ndarray) -> np.ndarray:
    return np.flip(arr, axis=1)

def _flip_v(arr: np.ndarray) -> np.ndarray:
    return np.flip(arr, axis=0)

# ---------------- core analyses (general) ----------------

def rb_stats(rulebooks: dict, outdir: Path):
    """Distribution of outputs per (phase,region)."""
    rules = rulebooks["rules"]
    phase_keys, regions = _phase_keys_and_regions(rules)
    rows = []
    for ph in phase_keys:
        for reg in regions:
            book = rules[ph][reg] if reg != "all" else rules[ph]
            # if reg=="all", book itself is the mapping
            if reg == "all":
                out_counts = Counter(book.values())
            else:
                out_counts = Counter(book.values())
            total = sum(out_counts.values())
            for y, c in out_counts.items():
                rows.append([int(ph), reg, int(y), c, c/total if total else 0.0])
    _save_csv(rows, Path(outdir) / "rb_output_distribution.csv",
              header=["phase","region","label_idx","count","fraction"])

def symmetry_checks(rulebooks: dict, sample=50000, outdir: Path | None = None):
    """
    Test orientation symmetry on a single (phase, region) rulebook.
    Strategy:
      - Prefer region='interior' if available (no OOB).
      - Otherwise, filter out keys that contain OOB (-1) so rotations stay comparable.
      - For each transform, count how many rotated keys exist and how often outputs agree.
    """
    rng = np.random.default_rng(123)
    rules = rulebooks["rules"]

    # choose phase 0 (or smallest), region preferring 'interior'
    phase_keys = sorted(rules.keys(), key=int)
    ph = phase_keys[0]
    regions = list(rules[ph].keys())
    if "interior" in regions:
        reg = "interior"
    else:
        # pick any region, but we'll filter OOB below
        reg = regions[0]

    book = rules[ph][reg]
    keys = list(book.keys())

    # helper: check if a key contains -1 (OOB)
    def has_oob(k: str) -> bool:
        return "-1" in k  # safe and fast

    # If not interior, filter out OOB-containing keys
    if reg != "interior":
        keys = [k for k in keys if not has_oob(k)]

    # sample down
    if len(keys) == 0:
        rows = [["transform","agree","tested","fraction_agree"],
                ["rot90", 0, 0, 0.0],
                ["flip_h",0, 0, 0.0],
                ["flip_v",0, 0, 0.0]]
        if outdir:
            save_csv(rows[1:], outdir / "rb_symmetry_checks.csv", header=rows[0])
        return rows[1:]

    if len(keys) > sample:
        keys = list(rng.choice(keys, size=sample, replace=False))

    # counters per transform
    tested = {"rot90": 0, "flip_h": 0, "flip_v": 0}
    agree  = {"rot90": 0, "flip_h": 0, "flip_v": 0}

    for k in keys:
        y0 = book[k]
        arr = _key_to_array(k, 5)
        var = {
            "rot90": _array_to_key(np.rot90(arr, 1)),
            "flip_h": _array_to_key(np.flip(arr, axis=1)),
            "flip_v": _array_to_key(np.flip(arr, axis=0)),
        }
        for name, k2 in var.items():
            y2 = book.get(k2, None)
            if y2 is None:
                continue
            tested[name] += 1
            agree[name]  += int(y2 == y0)

    rows = []
    for name in ["rot90","flip_h","flip_v"]:
        t = tested[name]
        a = agree[name]
        frac = (a / t) if t else 0.0
        rows.append([name, a, t, frac])

    if outdir:
        _save_csv(rows, outdir / "rb_symmetry_checks.csv",
                 header=["transform","agree","tested","fraction_agree"])
    return rows


def phase_delta(rulebooks: dict, outdir: Path):
    """For each region, among keys common to two phases, count output differences."""
    rules = rulebooks["rules"]
    phase_keys, regions = _phase_keys_and_regions(rules)
    if len(phase_keys) < 2:
        print("[phase_delta] only one phase present; skipping.")
        return
    p0, p1 = phase_keys[:2]
    rows = []
    for reg in regions:
        b0 = rules[p0][reg] if reg != "all" else rules[p0]
        b1 = rules[p1][reg] if reg != "all" else rules[p1]
        common = set(b0.keys()) & set(b1.keys())
        diff = sum(1 for k in common if b0[k] != b1[k])
        rows.append([reg, len(common), diff, diff/len(common) if common else 0.0])
    _save_csv(rows, Path(outdir) / "rb_phase_delta.csv",
              header=["region","common_keys","different_outputs","fraction_diff"])

def three_by_three_collapse(rulebooks: dict, outdir: Path):
    """
    Collapse each 5×5 key to its center 3×3 and count how often that induces conflicts
    (i.e., same 3×3 maps to multiple outputs). High conflict => true 5×5 dependence.
    """
    def collapse_to_3x3(k5: str) -> str:
        arr5 = _key_to_array(k5, 5)
        a3 = arr5[1:4, 1:4]
        return _array_to_key(a3)

    rules = rulebooks["rules"]
    phase_keys, regions = _phase_keys_and_regions(rules)

    rows = []
    for ph in phase_keys:
        for reg in regions:
            book = rules[ph][reg] if reg != "all" else rules[ph]
            buckets = defaultdict(list)
            for k5, y in book.items():
                buckets[collapse_to_3x3(k5)].append(y)
            total = len(book)
            conflicts = sum(1 for ys in buckets.values() if len(set(ys)) > 1)
            rows.append([int(ph), reg, total, conflicts, conflicts/total if total else 0.0])
    _save_csv(rows, Path(outdir) / "rb_collapse_3x3.csv",
              header=["phase","region","total_5x5_keys","conflicting_3x3_classes","fraction_conflicting"])

def position_saliency(rulebooks: dict, outdir: Path, sample=50000, mask_val=-999):
    """
    Occlusion test: mask each position and see how many masked patterns are ambiguous
    (i.e., map to >1 distinct outputs). Higher fraction ambiguous => position carries info.
    """
    rng = np.random.default_rng(7)
    rules = rulebooks["rules"]
    phase_keys, regions = _phase_keys_and_regions(rules)
    ph = phase_keys[0]; reg = regions[0]
    book = rules[ph][reg] if reg != "all" else rules[ph]
    keys = list(book.keys())
    if len(keys) > sample:
        keys = list(rng.choice(keys, size=sample, replace=False))

    def mask_at(arr, pos):
        a = arr.copy()
        i, j = divmod(pos, 5)
        a[i, j] = mask_val
        return a

    # Build masked->outputs per position
    masked_outputs = [defaultdict(set) for _ in range(25)]
    for k in keys:
        arr = _key_to_array(k, 5)
        y = book[k]
        for pos in range(25):
            mk = _array_to_key(mask_at(arr, pos))
            masked_outputs[pos][mk].add(y)

    rows = []
    for pos in range(25):
        amb = sum(1 for s in masked_outputs[pos].values() if len(s) > 1)
        tot = len(masked_outputs[pos])
        rows.append([pos, amb, tot, amb / max(1, tot)])
    _save_csv(rows, Path(outdir) / "rb_pos_saliency.csv",
              header=["pos_0to24","ambiguous_masked_classes","total_masked_classes","fraction_ambiguous"])

def mutual_info_positions(rulebooks: dict, outdir: Path, sample=150000):
    """
    Mutual information I(neighbor_position ; output) for a single (phase,region).
    """
    rules = rulebooks["rules"]
    phase_keys, regions = _phase_keys_and_regions(rules)
    ph = phase_keys[0]; reg = regions[0]
    book = rules[ph][reg] if reg != "all" else rules[ph]

    rng = np.random.default_rng(11)
    keys = list(book.keys())
    if len(keys) > sample:
        keys = list(rng.choice(keys, size=sample, replace=False))

    # collect joint counts per position
    joint = [Counter() for _ in range(25)]
    out_counts = Counter()
    for k in keys:
        arr = _key_to_array(k, 5)
        y = book[k]
        out_counts[y] += 1
        for pos in range(25):
            joint[pos][(int(arr.flat[pos]), y)] += 1

    N = len(keys)
    py = {y: c/N for y, c in out_counts.items()}
    rows = []
    for pos in range(25):
        # p(x), p(x,y)
        px = defaultdict(float)
        for (x, y), c in joint[pos].items():
            px[x] += c / N
        mi = 0.0
        for (x, y), c in joint[pos].items():
            pxy = c / N
            denom = (px[x] * py[y]) if (px[x] > 0 and py[y] > 0) else 1e-12
            mi += pxy * math.log2(max(pxy / denom, 1e-12))
        rows.append([pos, mi])
    _save_csv(rows, Path(outdir) / "rb_mutual_info_positions.csv",
              header=["pos_0to24","MI_bits"])

def train_tree_surrogate(rulebooks: dict, outdir: Path, phase: int | None = None,
                         region: str | None = None, max_depth=6, sample=100000):
    """
    Train a small DecisionTree to approximate the rulebook for (phase, region).
    Exports feature importances and a text dump of the tree.
    """
    rules = rulebooks["rules"]
    phase_keys, regions = _phase_keys_and_regions(rules)
    ph = str(phase) if phase is not None else phase_keys[0]
    reg = region if (region in regions) else regions[0]

    book = rules[ph][reg] if reg != "all" else rules[ph]
    rng = np.random.default_rng(5)
    keys = list(book.keys())
    if len(keys) > sample:
        keys = list(rng.choice(keys, size=sample, replace=False))

    X = []; y = []
    for k in keys:
        arr = _key_to_array(k, 5).reshape(-1)  # 25 ints (includes OOB value)
        X.append(arr.tolist())
        y.append(book[k])
    X = np.array(X, dtype=np.int16)
    y = np.array(y, dtype=np.int16)

    # train/test split
    n = len(X); m = max(1, int(0.8*n))
    idx = rng.permutation(n)
    tr, te = idx[:m], idx[m:]
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    clf.fit(X[tr], y[tr])
    acc = float((clf.predict(X[te]) == y[te]).mean()) if len(te) else 1.0

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    imp_path = outdir / "rb_tree_importances.csv"
    txt_path = outdir / "rb_tree_rules.txt"
    with open(imp_path, "w") as f:
        f.write("feature,importance\n")
        for i, imp in enumerate(clf.feature_importances_):
            f.write(f"pos{i},{imp}\n")
    with open(txt_path, "w") as f:
        f.write(export_text(clf, feature_names=[f"p{i}" for i in range(25)]))
        f.write(f"\n\n# held-out accuracy: {acc:.4f}\n")
    print(f"[saved] {imp_path}, {txt_path} (acc={acc:.4f})")

# ---------------- wrappers matching CLI ----------------

def rb_orientation_invariance(rulebooks: dict, outdir: Path):
    """Wrapper: run symmetry_checks and save CSV."""
    symmetry_checks(rulebooks, sample=50000, outdir=Path(outdir))

def rb_center_dependence(rulebooks: dict, outdir: Path):
    """
    Measure dependence on the center cell:
      - H(Y) overall vs H(Y|center)
      - also dump per-center output distribution
    """
    rules = rulebooks["rules"]
    phase_keys, regions = _phase_keys_and_regions(rules)
    ph = phase_keys[0]; reg = regions[0]
    book = rules[ph][reg] if reg != "all" else rules[ph]

    # parse center index = 12 (0..24, row 2 col 2)
    out_counts = Counter()
    by_center = defaultdict(Counter)
    for k, y in book.items():
        arr = _key_to_array(k, 5).reshape(-1)
        c = int(arr[12])
        y = int(y)
        out_counts[y] += 1
        by_center[c][y] += 1

    def H_from_counts(cnt: Counter) -> float:
        N = sum(cnt.values()) or 1
        return -sum((c/N) * math.log2(c/N) for c in cnt.values() if c)

    HY = H_from_counts(out_counts)
    # H(Y|C) = sum_c p(c) H(Y|c)
    N = sum(out_counts.values()) or 1
    all_center = Counter()
    for c, cnt in by_center.items():
        all_center[c] = sum(cnt.values())
    H_Y_given_C = 0.0
    rows = []
    for c, cnt in by_center.items():
        pc = (sum(cnt.values()) / N)
        Hyc = H_from_counts(cnt)
        H_Y_given_C += pc * Hyc
        for y, v in cnt.items():
            rows.append([c, y, v, v / max(1, sum(cnt.values()))])

    _save_csv(rows, Path(outdir) / "rb_center_output_dist.csv",
              header=["center_label","output_label","count","fraction"])
    _save_csv([[HY, H_Y_given_C, HY - H_Y_given_C]],
              Path(outdir) / "rb_center_dependence.csv",
              header=["H(Y)", "H(Y|center)", "I(Y; center)"])

def rb_positional_mi(rulebooks: dict, outdir: Path):
    """Wrapper: mutual information per 5×5 position."""
    mutual_info_positions(rulebooks, outdir=Path(outdir), sample=150000)

def rb_small_tree(rulebooks: dict, outdir: Path, phase: int = 0, region: str = "interior", max_depth: int = 3):
    """Wrapper: train a small decision tree surrogate."""
    train_tree_surrogate(rulebooks, outdir=Path(outdir), phase=phase, region=region, max_depth=max_depth)
