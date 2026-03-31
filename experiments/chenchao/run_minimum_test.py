from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.experiment_config import (
    DEVICE,
    INIT_SOURCE,
    CASP_MODE,
    DATASET_CONFIG,
    OPTIMISER_CONFIG,
    FOLD,
    EPOCHS,
)
from src.FuzzyInferenceSystem import FuzzyInferenceSystem
from src.trainable_fis import TrainableFIS
from src.fis_casps import decode_trapmf
from src.optimisation import OPTIMISERS

NON_GRADIENT_OPTIMISERS = {"pso", "ga", "de", "cmaes"}


def rmse(y, yhat):
    return torch.sqrt(torch.mean((y - yhat) ** 2))


def load_xy(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("@"):
                continue
            rows.append([float(x) for x in s.replace(",", " ").split()])
    arr = np.asarray(rows, dtype=np.float32)
    return arr[:, :-1], arr[:, -1]


def load_theta(path: str):
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


def load_rules(path: str, num_inputs: int):
    rules = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = [int(float(x)) for x in line.split()]
                r = r[: num_inputs + 1] + [1, 1]
                rules.append(r)
    return rules


def resolve_dataset_paths(base: Path, dataset: str, fold: int, seed: int, init_source: str, use_pca: bool):
    dataset_dir = base / dataset

    if use_pca:
        fold_dir = dataset_dir / f"{dataset}-pca" / "5-fold"
        tr = fold_dir / f"{dataset}-pca-5-{fold}tra.dat"
        te = fold_dir / f"{dataset}-pca-5-{fold}tst.dat"
    else:
        tr = dataset_dir / f"{dataset}-5-{fold}tra.dat"
        te = dataset_dir / f"{dataset}-5-{fold}tst.dat"

    if init_source == "kmeans":
        kmeans_dir = dataset_dir / "kmeans" / f"seed{seed}"
        theta_p = kmeans_dir / "theta.dat"
        rules_p = kmeans_dir / "rules.dat"
    else:
        theta_p = dataset_dir / "theta.dat"
        rules_p = dataset_dir / "rules.dat"

    return tr, te, theta_p, rules_p


def run_single(dataset: str, seed: int, optimiser: str) -> Tuple[float, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = DATASET_CONFIG[dataset]
    base = ROOT / "experiments" / "chenchao" / "data"

    tr_p, te_p, th_p, ru_p = resolve_dataset_paths(
        base, dataset, FOLD, seed, INIT_SOURCE, cfg["use_pca"]
    )

    Xtr_np, ytr_np = load_xy(str(tr_p))
    Xte_np, yte_np = load_xy(str(te_p))

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

    theta_init = load_theta(str(th_p))
    rules = load_rules(str(ru_p), Xtr_np.shape[1])

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

    opt_cfg = dict(OPTIMISER_CONFIG[optimiser])

    if cfg.get("theta_inputs_normalised", False):
        if optimiser in {"adam", "rmsprop", "sgd"}:
            opt_cfg["lr"] *= 0.05

    opt = OPTIMISERS[optimiser](
        model=model,
        loss_fn=rmse,
        device=DEVICE,
        **opt_cfg,
    )

    if optimiser in NON_GRADIENT_OPTIMISERS:
        opt.optimise(Xtr, ytr)
        train_rmse = float(opt.evaluate(Xtr, ytr))
    else:
        best = float("inf")
        best_state = opt.snapshot()

        for _ in range(EPOCHS):
            loss = opt.step(Xtr, ytr)
            val = float(loss.item())
            if val < best:
                best = val
                best_state = opt.snapshot()

        opt.restore(best_state)
        train_rmse = best

    test_rmse = float(opt.evaluate(Xte, yte))
    return train_rmse, test_rmse


def lookup_baseline(csv_path: Path, dataset: str, optimiser: str, seed: int):
    if not csv_path.exists():
        return None

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row.get("dataset") == dataset
                and row.get("optimiser") == optimiser
                and int(row.get("seed", -1)) == seed
            ):
                return float(row["train_rmse"]), float(row["test_rmse"])

    return None


def main():
    parser = argparse.ArgumentParser(description="Minimum test for adam+pso")
    parser.add_argument("--dataset", default="airfoil")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimisers", default="adam,pso")
    args = parser.parse_args()

    optimisers = [o.strip() for o in args.optimisers.split(",") if o.strip()]

    baseline_csv = ROOT / "experiments" / "chenchao" / "results" / "stage3_seed0_9_all_optimisers.csv"

    print("=" * 70)
    print("Minimum test")
    print("dataset:", args.dataset)
    print("seed   :", args.seed)
    print("optimisers:", optimisers)
    print("baseline:", baseline_csv)
    print("=" * 70)

    for opt in optimisers:
        print(f"\n[run] dataset={args.dataset} optimiser={opt} seed={args.seed}")
        train_rmse, test_rmse = run_single(args.dataset, args.seed, opt)
        print(f"computed train_rmse={train_rmse:.6f} test_rmse={test_rmse:.6f}")

        base = lookup_baseline(baseline_csv, args.dataset, opt, args.seed)
        if base is None:
            print("baseline not found in stage3 csv")
        else:
            btr, bte = base
            print(f"baseline train_rmse={btr:.6f} test_rmse={bte:.6f}")
            print(f"diff     train={train_rmse - btr:+.6f} test={test_rmse - bte:+.6f}")

    print("\n[DONE] minimum test finished.")


if __name__ == "__main__":
    main()
