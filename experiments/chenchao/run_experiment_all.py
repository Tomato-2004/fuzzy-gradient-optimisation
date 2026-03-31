from __future__ import annotations
import os, sys, csv
import numpy as np
import torch

# ======================================================
# Path
# ======================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ======================================================
# Config
# ======================================================
from experiments.experiment_config import *

from src.FuzzyInferenceSystem import FuzzyInferenceSystem
from src.trainable_fis import TrainableFIS
from src.fis_casps import decode_trapmf
from src.optimisation import OPTIMISERS

# ======================================================
# Constants
# ======================================================
NON_GRADIENT_OPTIMISERS = {"pso", "ga", "de", "cmaes"}

# ======================================================
# Utils (IDENTICAL)
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
# Experiment grid
# ======================================================
DATASETS = list(DATASET_CONFIG.keys())
OPTIMISERS_ALL = list(OPTIMISER_CONFIG.keys())
SEEDS = list(range(10))   # 10 seeds

# ======================================================
# Main
# ======================================================
def main():

    BASE = os.path.join(os.path.dirname(__file__), "data")

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "stage3_seed0_9_all_optimisers.csv")

    # ---------- 初始化 CSV（只在不存在时写表头）
    if not os.path.exists(out_csv):
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["dataset", "optimiser", "seed", "train_rmse", "test_rmse"],
            )
            writer.writeheader()

    run_id = 1
    total = len(DATASETS) * len(OPTIMISERS_ALL) * len(SEEDS)

    print("=" * 70)
    print(" Chen–Chao run_experiment_all (EXACT LOGIC)")
    print(" seeds     :", SEEDS)
    print(" datasets  :", DATASETS)
    print(" optimisers:", OPTIMISERS_ALL)
    print(" total     :", total)
    print("=" * 70)

    for DATASET in DATASETS:
        cfg = DATASET_CONFIG[DATASET]

        for OPTIMISER_NAME in OPTIMISERS_ALL:

            for SEED in SEEDS:

                print(
                    f"\n[{run_id}/{total}] "
                    f"DATASET={DATASET} OPTIMISER={OPTIMISER_NAME} SEED={SEED}"
                )
                run_id += 1

                torch.manual_seed(SEED)
                np.random.seed(SEED)

                tr_p, te_p, th_p, ru_p = resolve_dataset_paths(
                    BASE, DATASET, FOLD, SEED, INIT_SOURCE, cfg["use_pca"]
                )

                # ------------------------------
                # Load data
                # ------------------------------
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

                # ------------------------------
                # Load θ & rules
                # ------------------------------
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

                # ------------------------------
                # Build model
                # ------------------------------
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

                # ------------------------------
                # Optimiser
                # ------------------------------
                opt_cfg = dict(OPTIMISER_CONFIG[OPTIMISER_NAME])

                if cfg.get("theta_inputs_normalised", False):
                    if OPTIMISER_NAME in {"adam", "rmsprop", "sgd"}:
                        opt_cfg["lr"] *= 0.05

                opt = OPTIMISERS[OPTIMISER_NAME](
                    model=model,
                    loss_fn=rmse,
                    device=DEVICE,
                    **opt_cfg,
                )

                # ------------------------------
                # Train 
                # ------------------------------
                if OPTIMISER_NAME in NON_GRADIENT_OPTIMISERS:
                    opt.optimise(Xtr, ytr)
                    train_rmse = opt.evaluate(Xtr, ytr)
                else:
                    best = float("inf")
                    best_state = opt.snapshot()

                    for ep in range(1, EPOCHS + 1):
                        loss = opt.step(Xtr, ytr)
                        val = float(loss.item())

                        if val < best:
                            best = val
                            best_state = opt.snapshot()

                        if ep in LOG_EPOCHS:
                            print(f"[{ep:03d}] train={val:.6f}")

                    opt.restore(best_state)
                    train_rmse = best

                test_rmse = float(opt.evaluate(Xte, yte))

                print("Train RMSE:", train_rmse)
                print("Test  RMSE:", test_rmse)

                row = {
                    "dataset": DATASET,
                    "optimiser": OPTIMISER_NAME,
                    "seed": SEED,
                    "train_rmse": train_rmse,
                    "test_rmse": test_rmse,
                }

                with open(out_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["dataset", "optimiser", "seed", "train_rmse", "test_rmse"],
                    )
                    writer.writerow(row)

    print("\n[DONE] run_experiment_all finished.")
    print(f"[CSV] continuously saved to: {out_csv}")


if __name__ == "__main__":
    main()
