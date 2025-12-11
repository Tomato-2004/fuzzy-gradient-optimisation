import os
import sys
import csv
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



def run_single_dataset():
    ds_name = CONFIG["dataset"]
    print(f"\n=== Running SINGLE dataset: {ds_name} ===\n")

    # 1. load data
    X_train, X_test, y_train, y_test = load_dataset(ds_name)
    print(f"Loaded dataset: {ds_name} (train={len(X_train)}, test={len(X_test)})")

    # 2. build FIS
    init_cfg = CONFIG["initialisation"]
    fis = build_fis_from_config(X_train, y_train, init_cfg)
    print(f"FIS initialised with mf_method={init_cfg['mf_method']}, "
          f"rule_method={init_cfg['rule_method']}, "
          f"n_mf={init_cfg['n_mf']}, n_rules={init_cfg['n_rules']}")

    # 3. wrap model
    model = TrainableFIS(fis)

    # 4. choose optimiser
    opt_method = CONFIG["optimiser"]["method"]
    if opt_method not in OPTIMISERS:
        raise ValueError(f"Unknown optimiser: {opt_method}")

    print(f"Running optimiser: {opt_method}")
    model, history = OPTIMISERS[opt_method](
        model,
        X_train,
        y_train,
        CONFIG,
    )

    # 5. test eval
    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy().reshape(-1)
    mse = float(np.mean((preds - y_test) ** 2))

    print(f"\n[{ds_name}] Test MSE = {mse:.6f}\n")

    return {"dataset": ds_name, "optimiser": opt_method, "test_mse": mse}



def run_all_datasets():
    """
    跑所有数据集（一次只跑 config 指定的优化器）
    并输出 CSV。
    """

    opt_method = CONFIG["optimiser"]["method"]
    init_cfg = CONFIG["initialisation"]
    mf_method = init_cfg["mf_method"]

    print(f"\n=== Running ALL datasets with optimiser: {opt_method} (init={mf_method}) ===\n")

    results = []

    for ds_name in ALL_DATASETS:
        print("\n----------------------------------------")
        print(f"Dataset: {ds_name}")
        print("----------------------------------------")

        # 1. load data
        X_train, X_test, y_train, y_test = load_dataset(ds_name)
        print(f"Loaded dataset: {ds_name} (train={len(X_train)}, test={len(X_test)})")

        # 2. FIS initialisation
        fis = build_fis_from_config(X_train, y_train, init_cfg)
        print(f"FIS initialised with mf_method={init_cfg['mf_method']}, "
              f"rule_method={init_cfg['rule_method']}, "
              f"n_mf={init_cfg['n_mf']}, n_rules={init_cfg['n_rules']}")

        # 3. wrap model
        model = TrainableFIS(fis)

        # 4. run optimiser
        if opt_method not in OPTIMISERS:
            raise ValueError(f"Unknown optimiser: {opt_method}")

        print(f"Running optimiser: {opt_method}")
        model, history = OPTIMISERS[opt_method](
            model,
            X_train,
            y_train,
            CONFIG,
        )

        # 5. test eval
        model.eval()
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            preds = model(X_test_t).cpu().numpy().reshape(-1)

        mse = float(np.mean((preds - y_test) ** 2))
        print(f"[{ds_name}] Test MSE = {mse:.6f}")

        results.append({
            "dataset": ds_name,
            "optimiser": opt_method,
            "initialisation": mf_method,
            "test_mse": mse,
        })

    # ===== write CSV =====
    out_path = os.path.join(
        "experiments",
        f"results_all_{mf_method}_{opt_method}.csv"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "optimiser", "initialisation", "test_mse"])
        for r in results:
            writer.writerow([
                r["dataset"],
                r["optimiser"],
                r["initialisation"],
                r["test_mse"]
            ])

    print(f"\nFinished ALL datasets. Results saved to: {out_path}\n")



if __name__ == "__main__":
    # run_single_dataset()  
    run_all_datasets()
