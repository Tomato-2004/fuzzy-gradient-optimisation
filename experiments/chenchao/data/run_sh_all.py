from __future__ import annotations

import os
import sys
import math
import csv
import itertools
import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

# ======================================================
# Path (same as runexperiment)
# ======================================================
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ======================================================
# Import shared config / modules 
# ======================================================
from experiments.experiment_config import (
    DEVICE,
    INIT_SOURCE,
    CASP_MODE,
    DATASET_CONFIG,
    OPTIMISER_CONFIG,
)

from src.FuzzyInferenceSystem import FuzzyInferenceSystem
from src.trainable_fis import TrainableFIS
from src.fis_casps import decode_trapmf
from src.optimisation import OPTIMISERS

# ======================================================
# Constants
# ======================================================
NON_GRADIENT_OPTIMISERS = {"pso", "ga", "de", "cmaes"}

# ======================================================
# SH settings
# ======================================================
ROUND1_DATASETS = ["airfoil", "laser", "concrete"]
ROUND2_DATASETS = ["airfoil", "laser", "concrete", "ankara", "wine"]

ROUND1_SEEDS = [0, 1, 2]
ROUND2_SEEDS = [0]

ROUND1_EPOCHS = 100
ROUND2_EPOCHS = 300

ROUND1_POP = 10
ROUND2_POP = 30

ROUND1_ITERS = 10
ROUND2_ITERS = 10

ETA = 3
FINAL_KEEP = 2
MAX_CANDIDATES = 27

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# Optimiser candidate spaces 
# ======================================================
CANDIDATE_SPACE: Dict[str, Dict[str, List[Any]]] = {

    "adam": {
        "lr": [1e-4, 3e-4, 1e-3, 3e-3, 3e-2],
        "amsgrad": [True, False],
        "weight_decay": [0.0],
    },

    "rmsprop": {
        "lr": [1e-4, 3e-4, 1e-3],
        "alpha": [0.9, 0.99],
        "momentum": [0.0],
    },

    "sgd": {
        "lr": [1e-4, 3e-4, 1e-3, 3e-3],
        "momentum": [0.0, 0.9],
        "nesterov": [False],
    },

    "pso": {
        "w": [0.4, 0.72, 0.9],
        "c1": [1.2, 1.49, 2.0],
        "c2": [1.2, 1.49, 2.0],
    },

    "ga": {
        "mutation_rate": [0.05, 0.1, 0.2],
    },

    "de": {
        "F": [0.5, 0.8, 1.0],
        "CR": [0.7, 0.9],
    },

    "cmaes": {
        "sigma": [0.1, 0.3, 0.5],
    },
}

# ======================================================
# Utils 
# ======================================================
def rmse(y, yhat):
    return torch.sqrt(torch.mean((y - yhat) ** 2))


def load_xy(path):
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("@"):
                continue
            rows.append([float(x) for x in s.replace(",", " ").split()])
    arr = np.asarray(rows, dtype=np.float32)
    return arr[:, :-1], arr[:, -1]


def load_theta(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(
                    torch.tensor(
                        [float(x) for x in line.split()],
                        dtype=torch.float32,
                        device=DEVICE,
                    )
                )
    return out


def load_rules(path, num_inputs):
    rules = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = [int(float(x)) for x in line.split()]
                r = r[: num_inputs + 1] + [1, 1]
                rules.append(r)
    return rules


def resolve_dataset_paths(base, dataset, fold, seed, init_source, use_pca):
    dataset_dir = os.path.join(base, dataset)

    if use_pca:
        fold_dir = os.path.join(dataset_dir, f"{dataset}-pca", "5-fold")
        tr = os.path.join(fold_dir, f"{dataset}-pca-5-{fold}tra.dat")
        te = os.path.join(fold_dir, f"{dataset}-pca-5-{fold}tst.dat")
    else:
        tr = os.path.join(dataset_dir, f"{dataset}-5-{fold}tra.dat")
        te = os.path.join(dataset_dir, f"{dataset}-5-{fold}tst.dat")

    if init_source == "kmeans":
        kmeans_dir = os.path.join(dataset_dir, "kmeans", f"seed{seed}")
        theta_p = os.path.join(kmeans_dir, "theta.dat")
        rules_p = os.path.join(kmeans_dir, "rules.dat")
    else:
        theta_p = os.path.join(dataset_dir, "theta.dat")
        rules_p = os.path.join(dataset_dir, "rules.dat")

    return tr, te, theta_p, rules_p


# ======================================================
# run_single 
# ======================================================
def run_single(
    dataset: str,
    seed: int,
    optimiser: str,
    opt_cfg: Dict[str, Any],
    epochs: int,
    pop: int,
    iters: int,
) -> float:

    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = DATASET_CONFIG[dataset]
    BASE = ROOT / "experiments" / "chenchao" / "data"

    tr_p, te_p, th_p, ru_p = resolve_dataset_paths(
        BASE, dataset, fold=1, seed=seed,
        init_source=INIT_SOURCE,
        use_pca=cfg["use_pca"]
    )

    Xtr_np, ytr_np = load_xy(tr_p)
    Xte_np, yte_np = load_xy(te_p)

    if cfg.get("input_range_normalised", False):
        lo = Xtr_np.min(axis=0)
        hi = Xtr_np.max(axis=0)
        span = hi - lo
        span[span == 0.0] = 1.0
        Xtr_np = (Xtr_np - lo) / span
        Xte_np = (Xte_np - lo) / span

    Xtr = torch.tensor(Xtr_np, device=DEVICE)
    ytr = torch.tensor(ytr_np, device=DEVICE)
    Xte = torch.tensor(Xte_np, device=DEVICE)
    yte = torch.tensor(yte_np, device=DEVICE)

    theta_init = load_theta(th_p)
    rules = load_rules(ru_p, Xtr_np.shape[1])

    if cfg.get("theta_inputs_normalised", False):
        lo, hi = 0.05, 0.95
        for i in range(len(theta_init) - 1):
            th = theta_init[i]
            tmin, tmax = th.min(), th.max()
            if tmax > tmin:
                th = (th - tmin) / (tmax - tmin)
                th = th * (hi - lo) + lo
            else:
                th = torch.full_like(th, (lo + hi) / 2)
            theta_init[i] = th

    fis = FuzzyInferenceSystem(name="fis", device=DEVICE)

    model = TrainableFIS(
        fis=fis,
        theta_init=theta_init,
        rules=rules,
        num_mfs=cfg["num_mfs"],
        decode_trapmf_fn=decode_trapmf,
        casp_mode=CASP_MODE,
        device=DEVICE,
        theta_inputs_normalised=cfg["theta_inputs_normalised"],
    )

    local_cfg = copy.deepcopy(opt_cfg)

    if optimiser in NON_GRADIENT_OPTIMISERS:
        local_cfg["pop_size"] = pop
        local_cfg["iters"] = iters

    if cfg.get("theta_inputs_normalised", False):
        if optimiser in {"adam", "rmsprop", "sgd"}:
            local_cfg["lr"] *= 0.05

    opt = OPTIMISERS[optimiser](
        model=model,
        loss_fn=rmse,
        device=DEVICE,
        **local_cfg,
    )

    best = float("inf")
    best_state = opt.snapshot()

    if optimiser in NON_GRADIENT_OPTIMISERS:
        opt.optimise(Xtr, ytr)
        best = opt.evaluate(Xtr, ytr)
    else:
        for _ in range(epochs):
            loss = opt.step(Xtr, ytr)
            val = float(loss.item())
            if val < best:
                best = val
                best_state = opt.snapshot()
        opt.restore(best_state)

    return float(opt.evaluate(Xte, yte))


# ======================================================
# Baseline cache
# ======================================================
BASELINE_CACHE: Dict[str, Dict[Tuple[str, int, str], float]] = {}


def baseline_key(dataset: str, seed: int, round_tag: str):
    return (dataset, seed, round_tag)


def ensure_baselines(
    opt_name: str,
    datasets: List[str],
    seeds: List[int],
    round_tag: str,
    epochs: int,
    pop: int,
    iters: int,
):
    if opt_name not in BASELINE_CACHE:
        BASELINE_CACHE[opt_name] = {}

    base_cfg = copy.deepcopy(OPTIMISER_CONFIG[opt_name])

    for d in datasets:
        for s in seeds:
            k = baseline_key(d, s, round_tag)
            if k in BASELINE_CACHE[opt_name]:
                continue

            rmse0 = run_single(
                d, s, opt_name, base_cfg,
                epochs, pop, iters
            )
            BASELINE_CACHE[opt_name][k] = max(rmse0, 1e-12)


def normalised_score(
    opt_name: str,
    dataset: str,
    seed: int,
    round_tag: str,
    rmse_val: float,
) -> float:
    base = BASELINE_CACHE[opt_name][baseline_key(dataset, seed, round_tag)]
    return rmse_val / base


# ======================================================
# SH main
# ======================================================
def main():

    results = []

    for opt_name, space in CANDIDATE_SPACE.items():
        print(f"\n=== {opt_name.upper()} ===")

        opt_dir = OUT_DIR / opt_name
        opt_dir.mkdir(parents=True, exist_ok=True)

        keys = list(space.keys())
        all_cfgs = list(itertools.product(*space.values()))
        np.random.shuffle(all_cfgs)
        candidates = [dict(zip(keys, v)) for v in all_cfgs[:MAX_CANDIDATES]]

        # ---------- Round 1 ----------
        ensure_baselines(
            opt_name,
            ROUND1_DATASETS,
            ROUND1_SEEDS,
            "r1",
            ROUND1_EPOCHS,
            ROUND1_POP,
            ROUND1_ITERS,
        )

        r1 = []
        for cfg in candidates:
            scores = []
            for d in ROUND1_DATASETS:
                for s in ROUND1_SEEDS:
                    rmse_val = run_single(
                        d, s, opt_name, cfg,
                        ROUND1_EPOCHS, ROUND1_POP, ROUND1_ITERS
                    )
                    scores.append(
                        normalised_score(opt_name, d, s, "r1", rmse_val)
                    )

            r1.append((float(np.mean(scores)), cfg))

        r1.sort(key=lambda x: x[0])
        survivors = r1[: math.ceil(len(r1) / ETA)]

        # ---------- Round 2 ----------
        ensure_baselines(
            opt_name,
            ROUND2_DATASETS,
            ROUND2_SEEDS,
            "r2",
            ROUND2_EPOCHS,
            ROUND2_POP,
            ROUND2_ITERS,
        )

        round2_rows = []
        r2 = []

        for idx, (_, cfg) in enumerate(survivors):
            cfg_id = f"{opt_name}_{idx:02d}"
            row = {"cfg_id": cfg_id, **cfg}

            scores = []
            for d in ROUND2_DATASETS:
                rmse_val = run_single(
                    d, 0, opt_name, cfg,
                    ROUND2_EPOCHS, ROUND2_POP, ROUND2_ITERS
                )
                row[f"{d}_rmse"] = rmse_val
                scores.append(
                    normalised_score(opt_name, d, 0, "r2", rmse_val)
                )

            mean_score = float(np.mean(scores))
            r2.append((mean_score, cfg))
            round2_rows.append(row)

            print("[R2]", opt_name, cfg, mean_score)

        
        csv_path = opt_dir / "round2_seed0_rmse.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=round2_rows[0].keys())
            writer.writeheader()
            writer.writerows(round2_rows)

        r2.sort(key=lambda x: x[0])
        final = r2[:FINAL_KEEP]

        for rank, (score, cfg) in enumerate(final, 1):
            results.append({
                "optimiser": opt_name,
                "rank": rank,
                "norm_rmse": score,
                **cfg,
            })


if __name__ == "__main__":
    main()
