from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ======================================================
# Project root
# ======================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ======================================================
# Config registry ONLY
# ======================================================
from experiments.experiment_config import DATASET_CONFIG

EPS = 0.01


# ======================================================
# Robust .dat loader 
# ======================================================
def load_dat_matrix(path: str) -> np.ndarray:
    rows = []
    in_data = False

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

   
            if s.startswith("@"):
                if s.lower().startswith("@data"):
                    in_data = True
                continue

 
            if not in_data and ("@relation" in open(path, "r", encoding="utf-8", errors="ignore").read(2000).lower()):
                continue

            parts = s.replace(",", " ").split()
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"[genrule] No numeric rows parsed from: {path}")

    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"[genrule] Parsed matrix has invalid shape: {arr.shape} from {path}")
    return arr


# ======================================================
# KMeans 
# ======================================================
def kmeans_sorted_labels(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    uniq = np.unique(x)
    k_eff = min(int(k), int(len(uniq)))

    if k_eff <= 1:
        return np.ones_like(x, dtype=int)

    km = KMeans(
        n_clusters=k_eff,
        init="k-means++",
        max_iter=300,
        n_init=10,
        random_state=seed,
    )
    labels0 = km.fit_predict(x.reshape(-1, 1))

    present = np.unique(labels0)
    means = {c: float(np.mean(x[labels0 == c])) for c in present}
    order = sorted(present, key=lambda c: means[c])
    remap = {old: new + 1 for new, old in enumerate(order)}
    return np.array([remap[c] for c in labels0], dtype=int)


# ======================================================
# Utilities
# ======================================================
def category_ranges(df: pd.DataFrame, base_col: str, cat_col: str):
    out = []
    for c in sorted(df[cat_col].unique()):
        v = df.loc[df[cat_col] == c, base_col].to_numpy(dtype=float)
        out.append((float(np.min(v)), float(np.max(v))))
    return out


def most_common_output_and_count(x: pd.Series):
    mode = x.mode()
    if not mode.empty:
        m = int(mode.iloc[0])
        cnt = int((x == m).sum())
        return m, cnt
    return np.nan, 0


# ======================================================
# Theta initialisation (EXACT SEMANTICS FROM YOUR SINGLE)
# ======================================================
def init_theta_input(ranges, eps=EPS):
    m = len(ranges)
    n = (m - 1) * 4
    theta = np.zeros(n)

    theta[0] = ranges[0][0]
    theta[1] = (
        ranges[1][0] + (ranges[1][1] - ranges[1][0]) / 3 - eps
        if m > 2 else ranges[1][1] - eps
    )

    j = 0
    if m > 2:
        for j in range(1, m - 1):
            theta[j * 4 - 2] = (
                min(ranges[j - 1][0] + eps, ranges[j - 1][1] - eps)
                if j == 1
                else ranges[j - 1][1]
                - (ranges[j - 1][1] - ranges[j - 1][0]) / 3
                + eps
            )
            theta[j * 4 - 1] = ranges[j][0] + (ranges[j][1] - ranges[j][0]) / 3
            theta[j * 4] = ranges[j][1] - (ranges[j][1] - ranges[j][0]) / 3
            theta[j * 4 + 1] = (
                max(ranges[j + 1][1] - eps, ranges[j + 1][0] + eps)
                if j == m - 2
                else ranges[j + 1][0]
                + (ranges[j + 1][1] - ranges[j + 1][0]) / 3
                - eps
            )

    theta[n - 2] = (
        ranges[j][1] - (ranges[j][1] - ranges[j][0]) / 3 + eps
        if m > 2 else ranges[j][0] + eps
    )
    theta[n - 1] = ranges[j + 1][1]
    return theta


def init_theta_output(ranges, eps=EPS):
    m = len(ranges)
    n = (m - 1) * 4
    theta = np.zeros(n)

    theta[0] = ranges[0][0]
    theta[1] = (
        ranges[1][0] + (ranges[1][1] - ranges[1][0]) / 2 - eps * 2
        if m > 2 else ranges[1][1] - eps
    )

    j = 0
    if m > 2:
        for j in range(1, m - 1):
            theta[j * 4 - 2] = (
                min(ranges[j - 1][0] + eps, ranges[j - 1][1] - eps)
                if j == 1
                else ranges[j - 1][1]
                - (ranges[j - 1][1] - ranges[j - 1][0]) / 2
                + eps * 2
            )
            theta[j * 4 - 1] = ranges[j][0] + (ranges[j][1] - ranges[j][0]) / 2 - eps
            theta[j * 4] = ranges[j][1] - (ranges[j][1] - ranges[j][0]) / 2 + eps
            theta[j * 4 + 1] = (
                max(ranges[j + 1][1] - eps, ranges[j + 1][0] + eps)
                if j == m - 2
                else ranges[j + 1][0]
                + (ranges[j + 1][1] - ranges[j + 1][0]) / 2
                - eps * 2
            )

    theta[n - 2] = (
        ranges[j][1] - (ranges[j][1] - ranges[j][0]) / 2 + eps * 2
        if m > 2 else ranges[j][0] + eps
    )
    theta[n - 1] = ranges[j + 1][1]
    return theta


# ======================================================
# One run (dataset, seed) — EXACT SINGLE SEMANTICS
# ======================================================
def genrule_one(dataset: str, seed: int):
    cfg = DATASET_CONFIG[dataset]
    np.random.seed(seed)

    num_mfs = cfg["num_mfs"]

    this_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(this_dir, dataset)

    if cfg.get("use_pca", False):
        data_path = os.path.join(dataset_dir, f"{dataset}-pca", f"{dataset}-pca.dat")
    else:
        data_path = os.path.join(dataset_dir, f"{dataset}.dat")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[genrule] data not found: {data_path}")

    out_dir = os.path.join(dataset_dir, "kmeans", f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)
    rules_out = os.path.join(out_dir, "rules.dat")
    theta_out = os.path.join(out_dir, "theta.dat")
    num_mfs_out = os.path.join(out_dir, "num_mfs.json")

    arr = load_dat_matrix(data_path)
    num_inputs = arr.shape[1] - 1

    if len(num_mfs) != num_inputs + 1:
        raise ValueError(
            f"[genrule] NUM_MFS length mismatch: {dataset}\n"
            f"num_inputs={num_inputs}, num_mfs={num_mfs}"
        )

    cols = [f"Input{i+1}" for i in range(num_inputs)] + ["Output"]
    data = pd.DataFrame(arr, columns=cols)

    # input range normalisation (GENRULE stage)
    if cfg.get("input_range_normalised", False):
        input_cols = cols[:-1]
        lo = data[input_cols].min()
        hi = data[input_cols].max()
        span = hi - lo
        span[span == 0.0] = 1.0
        data[input_cols] = (data[input_cols] - lo) / span

    # KMeans categorisation for ALL cols (including Output)
    for i, c in enumerate(cols):
        data[f"{c}_cat"] = kmeans_sorted_labels(data[c].to_numpy(float), num_mfs[i], seed)

    # RULES (sorted)
    input_cat_cols = [f"{c}_cat" for c in cols[:-1]]
    total_n = len(data)

    rows = []
    for key, grp in data.groupby(input_cat_cols, observed=True)["Output_cat"]:
        m, cnt = most_common_output_and_count(grp)
        rows.append(list(key) + [m, cnt, len(grp), cnt / total_n * 100])

    rules_df = (
        pd.DataFrame(
            rows,
            columns=input_cat_cols + ["Most_Common_Output", "Most_Common_Output_Count", "Group_Size", "Coverage"],
        )
        .sort_values(by=input_cat_cols)
        .reset_index(drop=True)
    )
    rules_df.to_csv(rules_out, sep="\t", header=False, index=False)

    # THETA
    ranges = {c: category_ranges(data, c, f"{c}_cat") for c in cols}

    out_ranges = ranges["Output"]
    d0 = out_ranges[0][1] - out_ranges[0][0]
    dL = out_ranges[-1][1] - out_ranges[-1][0]
    out_ranges[0] = (out_ranges[0][0] - d0 / 2, out_ranges[0][1])
    out_ranges[-1] = (out_ranges[-1][0], out_ranges[-1][1] + dL / 2)
    ranges["Output"] = out_ranges

    with open(theta_out, "w", encoding="utf-8") as f:
        for c in cols[:-1]:
            f.write("\t".join(map(str, init_theta_input(ranges[c]))) + "\n")
        f.write("\t".join(map(str, init_theta_output(ranges["Output"]))) + "\n")

    # actual MF counts
    actual_num_mfs = [int(data[f"{c}_cat"].nunique()) for c in cols]
    with open(num_mfs_out, "w", encoding="utf-8") as f:
        json.dump(actual_num_mfs, f, indent=2)

    print(f"[DONE] {dataset} seed={seed}")
    print("requested NUM_MFS :", num_mfs)
    print("actual    NUM_MFS :", actual_num_mfs)


# ======================================================
# ALL datasets × ALL seeds
# ======================================================
def main():
    datasets = list(DATASET_CONFIG.keys())
    seeds = list(range(10))
    total = len(datasets) * len(seeds)
    k = 1

    print("=" * 60)
    print(" Chen–Chao KMeans rule/theta generation (ALL) — EXACT")
    print(" Datasets :", datasets)
    print(" Seeds    :", seeds)
    print(" Total    :", total)
    print("=" * 60)

    for ds in datasets:
        for sd in seeds:
            print(f"\n[{k}/{total}] dataset={ds}, seed={sd}")
            k += 1
            genrule_one(ds, sd)

    print("\n[DONE] All datasets × all seeds generated.")


if __name__ == "__main__":
    main()
