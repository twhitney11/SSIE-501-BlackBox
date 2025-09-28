# rbxplore.py
from __future__ import annotations
import json, math, itertools, gzip
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

# ---------------- IO ----------------

def load_rulebooks(path: str | Path) -> dict:
    p = Path(path)
    text = p.read_text()
    return json.loads(text)

def save_csv(rows, path: Path, header=None):
    df = pd.DataFrame(rows, columns=header if header else None)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[saved] {path}")

def get_phases_regions(rules: dict):
    """
    Normalize phase keys to str and return (phase_keys_str, region_names).
    Assumes structure: rules[phase][region] = { key5x5: label }
    """
    phase_keys = [str(k) for k in rules.keys()]
    sample = rules[phase_keys[0]]
    # if regions present:
    if isinstance(sample, dict):
        regions = list(sample.keys())
    else:
        regions = ["all"]
    return phase_keys, regions

# ------------- helpers for 5x5 ----------------

def key_to_array(key: str, size=5) -> np.ndarray:
    vals = np.array(list(map(int, key.split(","))), dtype=np.int16)
    return vals.reshape(size, size)

def array_to_key(arr: np.ndarray) -> str:
    return ",".join(map(str, arr.reshape(-1).tolist()))

def rotate90(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, k=1)

def flip_h(arr: np.ndarray) -> np.ndarray:
    return np.flip(arr, axis=1)

def flip_v(arr: np.ndarray) -> np.ndarray:
    return np.flip(arr, axis=0)

# ------------- core analyses ----------------

def rb_stats(rulebooks: dict, outdir: Path):
    meta = rulebooks["meta"]; rules = rulebooks["rules"]
    phase_keys, regions = get_phases_regions(rules)

    rows = []
    for ph in phase_keys:
        for reg in regions:
            book = rules[ph][reg]
            out_counts = Counter(book.values())
            total = sum(out_counts.values())
            for y, c in out_counts.items():
                rows.append([int(ph), reg, int(y), c, c/total if total else 0.0])
    save_csv(rows, outdir / "rb_output_distribution.csv",
             header=["phase","region","label_idx","count","fraction"])


def symmetry_checks(rulebooks: dict, sample=50000, outdir: Path | None = None):
    rng = np.random.default_rng(123)
    rules = rulebooks["rules"]
    phase_keys, regions = get_phases_regions(rules)

    ph0, reg0 = phase_keys[0], regions[0]
    base_keys = list(rules[ph0][reg0].keys())
    if len(base_keys) > sample:
        base_keys = list(rng.choice(base_keys, size=sample, replace=False))

    checks = Counter()
    for key in base_keys:
        arr = key_to_array(key, 5)
        variants = {
            "rot90": array_to_key(rotate90(arr)),
            "flip_h": array_to_key(flip_h(arr)),
            "flip_v": array_to_key(flip_v(arr)),
        }
        y0 = rules[ph0][reg0].get(key, None)
        for name, k2 in variants.items():
            y2 = rules[ph0][reg0].get(k2, None)
            if y0 is not None and y2 is not None:
                checks[name] += int(y0 == y2)

    rows = [[name, int(count), int(len(base_keys)), count/len(base_keys)] for name, count in checks.items()]
    if outdir:
        save_csv(rows, outdir / "rb_symmetry_checks.csv", header=["transform","agree","tested","fraction_agree"])
    return rows

def phase_delta(rulebooks: dict, outdir: Path):
    """Keys present in both phases, same region, with different outputs."""
    rules = rulebooks["rules"]
    p0, p1 = sorted(rules.keys(), key=int)[:2]
    regions = list(rules[p0].keys())
    rows = []
    for reg in regions:
        b0 = rules[p0][reg]; b1 = rules[p1][reg]
        common = set(b0.keys()) & set(b1.keys())
        diff = sum(1 for k in common if b0[k] != b1[k])
        rows.append([reg, len(common), diff, diff/len(common) if common else 0.0])
    save_csv(rows, outdir / "rb_phase_delta.csv", header=["region","common_keys","different_outputs","fraction_diff"])

def three_by_three_collapse(rulebooks: dict, outdir: Path):
    def collapse_to_3x3(k5: str) -> str:
        arr5 = key_to_array(k5, 5)
        a3 = arr5[1:4, 1:4]
        return array_to_key(a3)

    rules = rulebooks["rules"]
    phase_keys, regions = get_phases_regions(rules)

    rows = []
    for ph in phase_keys:
        for reg in regions:
            book = rules[ph][reg]
            buckets = defaultdict(list)
            for k5, y in book.items():
                buckets[collapse_to_3x3(k5)].append(y)
            total = len(book)
            conflicts = sum(1 for ys in buckets.values() if len(set(ys)) > 1)
            rows.append([int(ph), reg, total, conflicts, conflicts/total if total else 0.0])
    save_csv(rows, outdir / "rb_collapse_3x3.csv",
             header=["phase","region","total_5x5_keys","conflicting_3x3_classes","fraction_conflicting"])

def position_saliency(rulebooks: dict, outdir: Path, sample=50000):
    """
    Occlusion test: mask each position and see how many keys become ambiguous (multiple outputs share same masked pattern).
    Lower ambiguity -> position less critical; higher ambiguity -> position carries discriminating info.
    """
    rng = np.random.default_rng(7)
    rules = rulebooks["rules"]
    phase = sorted(rules.keys(), key=int)[0]
    reg = list(rules[phase].keys())[0]
    book = rules[phase][reg]
    keys = list(book.keys())
    if len(keys) > sample:
        keys = list(rng.choice(keys, size=sample, replace=False))

    def mask_at(arr, pos, mask_val=-999):
        a = arr.copy()
        i, j = divmod(pos, 5)
        a[i, j] = mask_val
        return a

    ambiguous_counts = np.zeros(25, dtype=np.int64)
    for k in keys:
        arr = key_to_array(k, 5)
        y = book[k]
        for pos in range(25):
            mk = array_to_key(mask_at(arr, pos))
            # Count how many distinct outputs share this masked pattern
            # Build on the fly (cheap map)
            # We can use a small cache but keep it simple:
            # We’ll consider ambiguity if another key maps to same masked pattern with different y
            # Quick approach: sample a few alternative keys
            # For accuracy, build a dict per pos once:
            pass

    # Efficient version: build masked->set(outputs) per position
    masked_outputs = [defaultdict(set) for _ in range(25)]
    for k in keys:
        arr = key_to_array(k, 5)
        y = book[k]
        for pos in range(25):
            mk = array_to_key(mask_at(arr, pos))
            masked_outputs[pos][mk].add(y)

    rows = []
    for pos in range(25):
        amb = sum(1 for s in masked_outputs[pos].values() if len(s) > 1)
        tot = len(masked_outputs[pos])
        rows.append([pos, amb, tot, amb/max(1, tot)])
    save_csv(rows, outdir / "rb_pos_saliency.csv",
             header=["pos_0to24","ambiguous_masked_classes","total_masked_classes","fraction_ambiguous"])

def mutual_info_positions(rulebooks: dict, outdir: Path, sample=150000):
    """
    Mutual information I(neighbor_position ; output).
    Crude, but useful: treat each position’s value as a categorical feature.
    """
    from math import log2

    rules = rulebooks["rules"]
    phase = sorted(rules.keys(), key=int)[0]
    reg = list(rules[phase].keys())[0]
    book = rules[phase][reg]
    rng = np.random.default_rng(11)
    keys = list(book.keys())
    if len(keys) > sample:
        keys = list(rng.choice(keys, size=sample, replace=False))

    # collect joint counts per position
    joint = [Counter() for _ in range(25)]
    out_counts = Counter()
    for k in keys:
        arr = key_to_array(k, 5)
        y = book[k]
        out_counts[y] += 1
        for pos in range(25):
            joint[pos][(int(arr.flat[pos]), y)] += 1

    N = len(keys)
    py = {y: c/N for y, c in out_counts.items()}
    rows = []
    for pos in range(25):
        # p(x), p(x,y)
        cx = Counter(x for (x,_y), c in joint[pos].items() for _ in [0] for __ in [c])  # not elegant, but fine
        px = defaultdict(float)
        for (x,y), c in joint[pos].items():
            px[x] += c/N
        mi = 0.0
        for (x,y), c in joint[pos].items():
            pxy = c / N
            mi += pxy * math.log2(pxy / (px[x] * py[y] + 1e-12) + 1e-12)
        rows.append([pos, mi])
    save_csv(rows, outdir / "rb_mutual_info_positions.csv", header=["pos_0to24","MI_bits"])

def train_tree_surrogate(rulebooks: dict, outdir: Path, max_depth=6, sample=100000):
    """
    Train a small DecisionTree to approximate the rulebook for one (phase,region).
    Exports feature importances (which 25 slots + center label matter) and accuracy on a held-out slice.
    """
    from sklearn.tree import DecisionTreeClassifier, export_text
    rules = rulebooks["rules"]
    phase = sorted(rules.keys(), key=int)[0]
    reg = list(rules[phase].keys())[0]
    book = rules[phase][reg]

    keys = list(book.keys())
    rng = np.random.default_rng(5)
    if len(keys) > sample:
        keys = list(rng.choice(keys, size=sample, replace=False))

    X = []; y = []
    for k in keys:
        arr = key_to_array(k, 5).reshape(-1)  # 25 ints (includes OOB value)
        X.append(arr.tolist())
        y.append(book[k])
    X = np.array(X, dtype=np.int16)
    y = np.array(y, dtype=np.int16)

    # train/test split
    n = len(X); m = int(0.8*n)
    idx = rng.permutation(n)
    tr, te = idx[:m], idx[m:]
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    clf.fit(X[tr], y[tr])
    acc = (clf.predict(X[te]) == y[te]).mean()

    # export
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(outdir)/"rb_tree_importances.csv","w") as f:
        f.write("feature,importance\n")
        for i, imp in enumerate(clf.feature_importances_):
            f.write(f"pos{i}, {imp}\n")
    with open(Path(outdir)/"rb_tree_rules.txt","w") as f:
        f.write(export_text(clf, feature_names=[f"p{i}" for i in range(25)]))
    print(f"[saved] {Path(outdir)/'rb_tree_importances.csv'}, acc={acc:.4f} (held-out)")
