import os
import sys
import csv
import copy
import random
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
from experiments.utils.datasets import ALL_DATASETS, load_dataset
from src.trainable_fis import TrainableFIS


# ======================================================
# Final Adam configurations (from SH)
# ======================================================

FINAL_CONFIGS = [
    {
        "initialisation": "heuristic",
        "lr": 3e-3,
        "batch_size": 64,
        "weight_decay": 0.0,
    },
    {
        "initialisation": "kmeans_mf",
        "lr": 3e-3,
        "batch_size": 64,
        "weight_decay": 1e-6,
    },
    {
        "initialisation": "kmeans_mf",
        "lr": 1e-2,
        "batch_size": 128,
        "weight_decay": 0.0,
    },
]

SEEDS = [0, 1, 2, 3, 4]


# ======================================================
# Reproducibility
# ======================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ======================================================
# Final validation
# ======================================================

def run_final_validation():

    results = []

    print("\n=== Running FINAL Adam Validation ===")
    print(f"Datasets     : {len(ALL_DATASETS)}")
    print(f"Seeds        : {SEEDS}")
    print(f"Configs      : {len(FINAL_CONFIGS)}")
    print("=====================================\n")

    for cfg_id, cfg_hp in enumerate(FINAL_CONFIGS):

        init_method = cfg_hp["initialisation"]
        lr = cfg_hp["lr"]
        batch_size = cfg_hp["batch_size"]
        weight_decay = cfg_hp["weight_decay"]

        print("\n=====================================")
        print(f"CONFIG {cfg_id+1}")
        print(f"Init   : {init_method}")
        print(f"Adam   : lr={lr}, batch={batch_size}, wd={weight_decay}")
        print("=====================================\n")

        for dataset in ALL_DATASETS:
            for seed in SEEDS:

                print("-------------------------------------")
                print(f"Dataset : {dataset}")
                print(f"Seed    : {seed}")
                print("-------------------------------------")

                set_seed(seed)

                # ---- load data ----
                X_train, X_test, y_train, y_test = load_dataset(dataset)

                # ---- build config ----
                cfg = copy.deepcopy(CONFIG)

                cfg["initialisation"]["mf_method"] = init_method
                cfg["optimiser"]["method"] = "adam"

                cfg["adam_params"]["num_epochs"] = 100
                cfg["adam_params"]["lr"] = lr
                cfg["adam_params"]["batch_size"] = batch_size
                cfg["adam_params"]["weight_decay"] = weight_decay

                # ---- build model ----
                fis = build_fis_from_config(X_train, y_train, cfg["initialisation"])
                model = TrainableFIS(fis)

                # ---- train ----
                model, _ = OPTIMISERS["adam"](
                    model,
                    X_train,
                    y_train,
                    cfg,
                )

                # ---- test ----
                model.eval()
                with torch.no_grad():
                    preds = model(
                        torch.tensor(X_test, dtype=torch.float32),
                        point_n=cfg["adam_params"]["point_n"],
                    ).cpu().numpy().reshape(-1)

                mse = float(np.mean((preds - y_test) ** 2))
                print(f"[RESULT] Test MSE = {mse:.6f}")

                results.append({
                    "dataset": dataset,
                    "seed": seed,
                    "initialisation": init_method,
                    "optimiser": "adam",
                    "lr": lr,
                    "batch_size": batch_size,
                    "weight_decay": weight_decay,
                    "test_mse": mse,
                })

    # ======================================================
    # Write CSV
    # ======================================================

    out_dir = "experiments"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results_adam_final_validation.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "seed",
                "initialisation",
                "optimiser",
                "lr",
                "batch_size",
                "weight_decay",
                "test_mse",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n=====================================")
    print("FINAL VALIDATION FINISHED")
    print(f"Results saved to: {out_path}")
    print("=====================================\n")


# ======================================================
# Entry
# ======================================================

if __name__ == "__main__":
    run_final_validation()
