from __future__ import annotations

import os
import json
import math
import random
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# ======================================================
# GLOBAL SETTINGS
# ======================================================
SEED = 0
RANDOM_BUDGET = 5000

MF_MIN = 3
MF_MAX = 9
OUT_MF_MIN = 3
OUT_MF_MAX = 9

TOPK_RULES = 150
TARGET_COVERAGE = 78.0

MAX_DOMINANT_RULE = 6.0
MIN_EFFECTIVE_RULES = 30.0
MIN_MF_USAGE_ENTROPY = 0.55

W_LB_RMSE = 50.0
W_COV_TARGET = 2.0
W_RULES_TO_TARGET = 0.5
W_COV_TOPK = 0.5

W_RULE_ENT = 6.0
W_EFF_RULES = 0.6
W_MF_BAL = 6.0
W_DOM = 10.0
W_RULES = 0.05

NORMALIZE_INPUTS = True
EPS = 1e-12

np.random.seed(SEED)
random.seed(SEED)


# ======================================================
# DATASETS
# ======================================================
DATASETS_NO_PCA = [
    # "airfoil",
    # "autompg6",
    # "laser",
]

DATASETS_PCA = [
    "baseball",
    # "abalone",
    # "concrete",
    # "ankara",
    # "izmir",
    # "treasury",
    # "wine",
]


# ======================================================
# Utils
# ======================================================
def load_dat(path: str) -> pd.DataFrame:
    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("@") or line.startswith("%"):
                continue
            lines.append(line)

    sep = "," if "," in lines[0] else r"\s+"
    df = pd.read_csv(StringIO("\n".join(lines)), header=None, sep=sep)

    return df.astype(float)


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + EPS))) if p.size else 0.0


def kmeans_1d_sorted(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    uniq = np.unique(x)
    k = min(k, len(uniq))
    if k <= 1:
        return np.ones_like(x, dtype=int)

    km = KMeans(
        n_clusters=k,
        n_init=10,
        max_iter=300,
        random_state=SEED,
    )
    labels0 = km.fit_predict(x.reshape(-1, 1))

    centers = [(c, x[labels0 == c].mean()) for c in np.unique(labels0)]
    centers.sort(key=lambda t: t[1])
    remap = {old: i + 1 for i, (old, _) in enumerate(centers)}

    return np.array([remap[l] for l in labels0], dtype=int)


def teacher_coverage_metrics(rule_covs: np.ndarray) -> Dict[str, Any]:
    cov_sorted = np.sort(rule_covs)[::-1]
    cum_cov = np.cumsum(cov_sorted)

    k = min(TOPK_RULES, len(cum_cov))
    coverage_topK = float(cum_cov[k - 1]) if k else 0.0

    hit = np.where(cum_cov >= TARGET_COVERAGE)[0]
    if len(hit) == 0:
        return {
            "coverage_topK": coverage_topK,
            "rules_to_target": 10**9,
            "coverage_at_target": float(cum_cov[-1]),
            "cum_cov_all": float(cum_cov[-1]),
        }

    idx = int(hit[0])
    return {
        "coverage_topK": coverage_topK,
        "rules_to_target": idx + 1,
        "coverage_at_target": float(cum_cov[idx]),
        "cum_cov_all": float(cum_cov[-1]),
    }


@dataclass(frozen=True)
class MFConfig:
    mf_inputs: Tuple[int, ...]
    mf_output: int


# ======================================================
# Core evaluation 
# ======================================================
def evaluate_config(data: pd.DataFrame, cfg: MFConfig) -> Dict[str, Any]:
    df = data.copy()
    d = df.shape[1] - 1
    N = len(df)

    for i in range(d):
        df[f"I{i}_cat"] = kmeans_1d_sorted(df.iloc[:, i].to_numpy(), cfg.mf_inputs[i])

    df["O_cat"] = kmeans_1d_sorted(df.iloc[:, -1].to_numpy(), cfg.mf_output)

    rule_covs = []
    rule_inputs = []
    lb_mse = 0.0

    for key, grp in df.groupby([f"I{i}_cat" for i in range(d)], observed=True):
        n_c = len(grp)
        if n_c <= 1:
            continue

        y = grp.iloc[:, -1].to_numpy()
        lb_mse += (n_c / N) * np.var(y)

        cnt = grp["O_cat"].value_counts().iloc[0]
        rule_covs.append(cnt / N * 100.0)
        rule_inputs.append(key)

    if not rule_covs:
        return {"score": -1e18}

    rule_covs = np.asarray(rule_covs, dtype=float)
    lb_rmse = math.sqrt(lb_mse)

    covm = teacher_coverage_metrics(rule_covs)

    p = rule_covs / rule_covs.sum()
    H = entropy(p)
    eff_rules = math.exp(H)

    mf_bal = []
    for i in range(d):
        counts = pd.Series([r[i] for r in rule_inputs]).value_counts(normalize=True)
        mf_bal.append(entropy(counts.to_numpy()) / math.log(len(counts) + EPS))
    mf_balance = float(np.mean(mf_bal))

    if (
        rule_covs.max() > MAX_DOMINANT_RULE + 5
        or eff_rules < MIN_EFFECTIVE_RULES
        or mf_balance < MIN_MF_USAGE_ENTROPY
    ):
        return {"score": -1e18}

    score = (
        -W_LB_RMSE * lb_rmse
        + W_COV_TARGET * covm["coverage_at_target"]
        + W_COV_TOPK * covm["coverage_topK"]
        - W_RULES_TO_TARGET * min(covm["rules_to_target"], 1e8)
        + W_RULE_ENT * (H / math.log(len(rule_covs) + EPS))
        + W_EFF_RULES * eff_rules
        + W_MF_BAL * mf_balance
        - W_DOM * max(0.0, rule_covs.max() - MAX_DOMINANT_RULE)
        - W_RULES * len(rule_covs)
    )

    return {
        "mf_inputs": cfg.mf_inputs,
        "mf_output": cfg.mf_output,
        "lb_rmse": lb_rmse,
        "score": score,
        **covm,
    }


# ======================================================
# Main loop
# ======================================================
def run_all():
    base = os.path.dirname(os.path.abspath(__file__))

    for name in DATASETS_NO_PCA + DATASETS_PCA:
        print(f"\n===== DATASET: {name} =====")

        if name in DATASETS_NO_PCA:
            data_path = os.path.join(base, name, f"{name}.dat")
        else:
            data_path = os.path.join(base, name, f"{name}-pca", f"{name}-pca.dat")

        print("[DATA]", data_path)

        data = load_dat(data_path)

        if NORMALIZE_INPUTS:
            scaler = MinMaxScaler()
            data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

        d = data.shape[1] - 1

        out_dir = os.path.join(base, name, "rule_search_lb_rmse_teacher_cov")
        os.makedirs(out_dir, exist_ok=True)

        rows = []
        best = None
        seen = set()

        for _ in range(RANDOM_BUDGET):
            cfg = MFConfig(
                tuple(random.randint(MF_MIN, MF_MAX) for _ in range(d)),
                random.randint(OUT_MF_MIN, OUT_MF_MAX),
            )
            if cfg in seen:
                continue
            seen.add(cfg)

            r = evaluate_config(data, cfg)
            if "score" not in r:
                continue

            rows.append(r)
            if best is None or r["score"] > best["score"]:
                best = r

        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, "all_results.csv"),
            index=False,
        )
        with open(os.path.join(out_dir, "best.json"), "w") as f:
            json.dump(best, f, indent=2)

        print("[BEST]", best)


if __name__ == "__main__":
    run_all()
