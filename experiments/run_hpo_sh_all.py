import os
import sys
import csv
import random
import numpy as np
import torch
import copy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.experiment_config import CONFIG
from experiments.initialisation_registry import build_fis_from_config
from experiments.optimiser_registry import OPTIMISERS
from experiments.utils.datasets import load_dataset
from src.trainable_fis import TrainableFIS

# =========================
# HPO setting
# =========================

REPRESENTATIVE_DATASETS = [
    "Airfoil", "Concrete", "Energy",
    "Abalone", "Yacht", "Wine"
]

SEEDS = [0, 1, 2]
EPOCH_SCHEDULE = [20, 50, 100]
TOP_K = 2

OPTIMISERS_TO_RUN = [
    "adam", "sgd", "rmsprop",
    "pso", "ga", "de", "cmaes"
]

# =========================
# Search spaces
# =========================

SEARCH_SPACES = {
    "adam": {
        "lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "batch_size": [32, 64, 128],
        "weight_decay": [0.0, 1e-6, 1e-5],
    },
    "sgd": {
        "lr": [1e-3, 3e-3, 1e-2, 3e-2],
        "momentum": [0.0, 0.8, 0.9, 0.95],
        "batch_size": [32, 64, 128],
        "weight_decay": [0.0, 1e-6, 1e-5],
    },
    "rmsprop": {
        "lr": [1e-4, 3e-4, 1e-3, 3e-3],
        "alpha": [0.9, 0.95, 0.99],
        "batch_size": [32, 64, 128],
        "weight_decay": [0.0, 1e-6],
    },
    "pso": {
        "swarm_size": [20, 40, 60],
    },
    "ga": {
        "pop_size": [30, 50, 80],
        "mutation_rate": [0.05, 0.1, 0.2],
        "crossover_rate": [0.6, 0.8],
    },
    "de": {
        "F": [0.3, 0.5, 0.8],
        "CR": [0.7, 0.9],
    },
    "cmaes": {
        "sigma_init": [0.05, 0.1, 0.2],
        "population": [10, 20],
    },
}

# =========================
# Utilities
# =========================

def sample_configs(space, n=6):
    keys = list(space.keys())
    configs = []
    for _ in range(n):
        cfg = {k: random.choice(space[k]) for k in keys}
        configs.append(cfg)
    return configs


def split_train_val(X, y, val_ratio=0.2, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)

    split = int(len(X) * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]

    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def evaluate_config(opt_method, init_method, hp, epochs):
    scores = []

    for ds in REPRESENTATIVE_DATASETS:
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            X_train_full, X_test, y_train_full, y_test = load_dataset(ds)

            # 🔑 只在 train 中切 validation
            X_train, X_val, y_train, y_val = split_train_val(
                X_train_full, y_train_full, val_ratio=0.2, seed=seed
            )

            cfg = copy.deepcopy(CONFIG)
            cfg["optimiser"]["method"] = opt_method

            param_key = f"{opt_method}_params"
            cfg[param_key].update(hp)
            cfg[param_key]["num_epochs"] = epochs

            fis = build_fis_from_config(X_train, y_train, cfg["initialisation"])
            model = TrainableFIS(fis)

            model, _ = OPTIMISERS[opt_method](model, X_train, y_train, cfg)

            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_val, dtype=torch.float32)).cpu().numpy().reshape(-1)

            mse = float(np.mean((preds - y_val) ** 2))
            scores.append(mse)

    return float(np.median(scores))


def run_sh_for_optimiser(opt_method, init_method="kmeans_mf"):
    print(f"\n=== SH for optimiser: {opt_method} ===")

    candidates = sample_configs(SEARCH_SPACES[opt_method], n=6)

    for stage, epochs in enumerate(EPOCH_SCHEDULE):
        print(f"\nStage {stage+1}: epochs={epochs}")
        results = []

        for hp in candidates:
            score = evaluate_config(opt_method, init_method, hp, epochs)
            results.append((hp, score))
            print(f"  {hp} → median VAL MSE = {score:.5f}")

        results.sort(key=lambda x: x[1])
        candidates = [r[0] for r in results[:max(1, len(results)//2)]]

    return [(opt_method, init_method, hp, score) for hp, score in results]


def main():
    out_path = "experiments/results_hpo_sh_all.csv"
    os.makedirs("experiments", exist_ok=True)

    rows = []

    for opt in OPTIMISERS_TO_RUN:
        rows.extend(run_sh_for_optimiser(opt))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["optimiser", "initialisation", "hyperparams", "median_mse"])
        for r in rows:
            writer.writerow(r)

    print(f"\nSaved HPO results to {out_path}")


if __name__ == "__main__":
    main()
