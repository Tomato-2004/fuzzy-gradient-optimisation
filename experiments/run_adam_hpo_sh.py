import os
import sys
import csv
import random
import copy
import numpy as np
import torch

# ======================================================
# Path setup
# ======================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.experiment_config import CONFIG
from experiments.initialisation_registry import build_fis_from_config
from experiments.optimiser_registry import OPTIMISERS
from experiments.utils.datasets import load_dataset
from src.trainable_fis import TrainableFIS


# ======================================================
# HPO settings
# ======================================================

# 6 representative datasets for HPO
HPO_DATASETS = [
    "Airfoil",
    "Energy",
    "Concrete",
    "Yacht",
    "AutoMPG6",
    "Abalone",
]

SEEDS = [0, 1, 2]  # fewer seeds for HPO ranking

# Successive Halving schedule
EPOCH_SCHEDULE = [20, 50, 100]
KEEP_RATIO = [1/3, 1/2]  # after round 0, after round 1

# Number of initial random configs
N_INIT = 24


# ======================================================
# Adam hyperparameter search space
# ======================================================

LR_SPACE = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
BATCH_SPACE = [32, 64, 128]
WD_SPACE = [0.0, 1e-6, 1e-5]


def sample_adam_params():
    return {
        "lr": random.choice(LR_SPACE),
        "batch_size": random.choice(BATCH_SPACE),
        "weight_decay": random.choice(WD_SPACE),
    }


# ======================================================
# Reproducibility
# ======================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================
# Evaluate one hyperparameter config
# ======================================================

def evaluate_config(init_method, adam_params, num_epochs):
    """
    Return median test MSE across datasets,
    where each dataset MSE is averaged over seeds.
    """
    dataset_scores = []

    for ds in HPO_DATASETS:
        seed_scores = []

        for seed in SEEDS:
            set_seed(seed)

            X_train, X_test, y_train, y_test = load_dataset(ds)

            cfg = copy.deepcopy(CONFIG)
            cfg["initialisation"]["mf_method"] = init_method
            cfg["optimiser"]["method"] = "adam"

            cfg["adam_params"]["num_epochs"] = num_epochs
            cfg["adam_params"]["lr"] = adam_params["lr"]
            cfg["adam_params"]["batch_size"] = adam_params["batch_size"]
            cfg["adam_params"]["weight_decay"] = adam_params["weight_decay"]

            fis = build_fis_from_config(X_train, y_train, cfg["initialisation"])
            model = TrainableFIS(fis)

            model, _ = OPTIMISERS["adam"](model, X_train, y_train, cfg)

            model.eval()
            with torch.no_grad():
                preds = model(
                    torch.tensor(X_test, dtype=torch.float32),
                    point_n=cfg["adam_params"]["point_n"],
                ).cpu().numpy().reshape(-1)

            mse = float(np.mean((preds - y_test) ** 2))
            seed_scores.append(mse)

        dataset_scores.append(np.mean(seed_scores))

    return float(np.median(dataset_scores))


# ======================================================
# Successive Halving
# ======================================================

def run_sh_for_initialisation(init_method):
    print(f"\n=== SH for init method: {init_method} ===")

    # ----- Round 0 -----
    candidates = [sample_adam_params() for _ in range(N_INIT)]
    scores = []

    for hp in candidates:
        score = evaluate_config(init_method, hp, num_epochs=EPOCH_SCHEDULE[0])
        scores.append((hp, score))
        print(f"[Round 0] {hp} -> median MSE = {score:.6f}")

    scores.sort(key=lambda x: x[1])
    k1 = int(len(scores) * KEEP_RATIO[0])
    scores = scores[:k1]

    # ----- Round 1 -----
    new_scores = []
    for hp, _ in scores:
        score = evaluate_config(init_method, hp, num_epochs=EPOCH_SCHEDULE[1])
        new_scores.append((hp, score))
        print(f"[Round 1] {hp} -> median MSE = {score:.6f}")

    new_scores.sort(key=lambda x: x[1])
    k2 = int(len(new_scores) * KEEP_RATIO[1])
    scores = new_scores[:k2]

    # ----- Round 2 -----
    final_scores = []
    for hp, _ in scores:
        score = evaluate_config(init_method, hp, num_epochs=EPOCH_SCHEDULE[2])
        final_scores.append((hp, score))
        print(f"[Round 2] {hp} -> median MSE = {score:.6f}")

    final_scores.sort(key=lambda x: x[1])
    return final_scores


# ======================================================
# Main
# ======================================================

def main():
    os.makedirs("experiments", exist_ok=True)
    out_path = "experiments/results_adam_hpo_sh.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "initialisation",
            "lr",
            "batch_size",
            "weight_decay",
            "median_mse",
        ])

        for init_method in ["heuristic", "kmeans_mf"]:
            final_scores = run_sh_for_initialisation(init_method)
            for hp, score in final_scores:
                writer.writerow([
                    init_method,
                    hp["lr"],
                    hp["batch_size"],
                    hp["weight_decay"],
                    score,
                ])

    print("\n====================================")
    print("Adam HPO with Successive Halving finished")
    print(f"Results saved to: {out_path}")
    print("====================================\n")


if __name__ == "__main__":
    main()
