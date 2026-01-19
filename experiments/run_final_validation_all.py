import os
import sys
import csv
import time
import json
import copy
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.experiment_config import CONFIG
from experiments.initialisation_registry import build_fis_from_config
from experiments.optimiser_registry import OPTIMISERS
from experiments.utils.datasets import ALL_DATASETS, load_dataset
from src.trainable_fis import TrainableFIS

SEEDS = [0, 1, 2, 3, 4]


def load_best_configs(csv_path):
    best = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            opt = row["optimiser"]
            hp = json.loads(row["hyperparams"].replace("'", '"'))
            mse = float(row["median_mse"])

            if opt not in best or mse < best[opt]["mse"]:
                best[opt] = {"hp": hp, "mse": mse}
    return best


def main():
    best_cfgs = load_best_configs("experiments/results_hpo_sh_all.csv")
    out_path = "experiments/results_final_validation_all.csv"

    rows = []

    for opt, info in best_cfgs.items():
        for ds in ALL_DATASETS:
            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                X_train, X_test, y_train, y_test = load_dataset(ds)

                cfg = copy.deepcopy(CONFIG)
                cfg["optimiser"]["method"] = opt
                cfg[f"{opt}_params"].update(info["hp"])

                fis = build_fis_from_config(X_train, y_train, cfg["initialisation"])
                model = TrainableFIS(fis)

                t0 = time.perf_counter()
                model, _ = OPTIMISERS[opt](model, X_train, y_train, cfg)
                elapsed = time.perf_counter() - t0

                model.eval()
                with torch.no_grad():
                    preds = model(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy().reshape(-1)

                mse = float(np.mean((preds - y_test) ** 2))

                rows.append({
                    "dataset": ds,
                    "optimiser": opt,
                    "seed": seed,
                    "test_mse": mse,
                    "train_time": elapsed
                })

                print(f"{opt} | {ds} | seed={seed} → mse={mse:.5f}, time={elapsed:.2f}s")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "optimiser", "seed", "test_mse", "train_time"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nFinal results saved to {out_path}")


if __name__ == "__main__":
    main()
