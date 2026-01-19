import os
import sys
import csv
import random
import numpy as np
import torch
import itertools
import copy

# ======================================================
# Path setup
# ======================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.experiment_config import CONFIG
from experiments.initialisation_registry import build_fis_from_config
from experiments.optimiser_registry import OPTIMISERS
from experiments.utils.datasets import ALL_DATASETS, load_dataset
from src.trainable_fis import TrainableFIS


MF_METHODS = [
    "heuristic",
    "casp_single",
    "casp_adapt",
    "random_gauss",
    "kmeans_mf",
]

OPTIMISER_METHODS = [
    "adam",
    "pso",
    "ga",
    "de",
    "cmaes",
    "sgd",
    "rmsprop",
]

SEEDS = [0, 1, 2, 3, 4]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_grid_experiments():
    results = []

    print("\n=== Running FULL GRID EXPERIMENTS (STABLE) ===")
    print(f"MF methods   : {MF_METHODS}")
    print(f"Optimisers   : {OPTIMISER_METHODS}")
    print(f"Seeds        : {SEEDS}")
    print(f"Datasets     : {len(ALL_DATASETS)}")
    print("=============================================\n")

    for dataset, mf_method, opt_method, seed in itertools.product(
        ALL_DATASETS, MF_METHODS, OPTIMISER_METHODS, SEEDS
    ):
        print("\n----------------------------------------")
        print(f"Dataset      : {dataset}")
        print(f"MF init      : {mf_method}")
        print(f"Optimiser    : {opt_method}")
        print(f"Seed         : {seed}")
        print("----------------------------------------")

        set_seed(seed)

        X_train, X_test, y_train, y_test = load_dataset(dataset)

        cfg = copy.deepcopy(CONFIG)
        cfg["initialisation"] = {
            "mf_method": mf_method,
            "rule_method": "kmeans",
            "n_mf": CONFIG["initialisation"]["n_mf"],
            "n_rules": CONFIG["initialisation"]["n_rules"],
        }
        cfg["optimiser"] = {"method": opt_method}

        fis = build_fis_from_config(X_train, y_train, cfg["initialisation"])
        model = TrainableFIS(fis)

        if opt_method not in OPTIMISERS:
            raise ValueError(f"Unknown optimiser: {opt_method}")

        model, _ = OPTIMISERS[opt_method](model, X_train, y_train, cfg)

        # ---- test evaluation (explicit point_n for protocol consistency) ----
        model.eval()
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        point_n = (
            cfg.get("adam_params", {}).get("point_n", 101)
        )

        with torch.no_grad():
            preds = model(X_test_t, point_n=point_n).cpu().numpy().reshape(-1)

        mse = float(np.mean((preds - y_test) ** 2))
        if not np.isfinite(mse):
            print(f"[WARN] Non-finite MSE detected: {mse}")

        print(f"[RESULT] Test MSE = {mse:.6f}")

        results.append({
            "dataset": dataset,
            "optimiser": opt_method,
            "initialisation": mf_method,
            "seed": seed,
            "test_mse": mse,
        })

    # Write CSV
    os.makedirs("experiments", exist_ok=True)
    out_path = os.path.join("experiments", "results_full_grid_stable.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "optimiser", "initialisation", "seed", "test_mse"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n====================================")
    print("ALL EXPERIMENTS FINISHED")
    print(f"Results saved to: {out_path}")
    print("====================================\n")


if __name__ == "__main__":
    run_grid_experiments()
