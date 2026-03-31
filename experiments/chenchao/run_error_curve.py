from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

DATASETS = ["airfoil", "laser", "concrete"]
OPTIMISERS_TO_RUN = ["adam", "pso"]
SEEDS = [0, 1, 2]


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


def prepare_run(dataset: str, seed: int):
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

    return model, Xtr, ytr, Xte, yte, cfg


def run_adam_curve(dataset: str, seed: int):
    model, Xtr, ytr, Xte, yte, cfg = prepare_run(dataset, seed)

    opt_cfg = dict(OPTIMISER_CONFIG["adam"])
    if cfg.get("theta_inputs_normalised", False):
        opt_cfg["lr"] *= 0.05

    opt = OPTIMISERS["adam"](
        model=model,
        loss_fn=rmse,
        device=DEVICE,
        **opt_cfg,
    )

    best = float("inf")
    best_state = opt.snapshot()

    rows: List[Dict[str, float]] = []

    for ep in range(1, EPOCHS + 1):
        loss = opt.step(Xtr, ytr)
        train_rmse = float(loss.item())

        if train_rmse < best:
            best = train_rmse
            best_state = opt.snapshot()

        test_rmse = opt.evaluate(Xte, yte)

        rows.append(
            {
                "step": ep,
                "step_type": "epoch",
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
            }
        )

    opt.restore(best_state)
    return rows


def run_pso_curve(dataset: str, seed: int):
    model, Xtr, ytr, Xte, yte, _cfg = prepare_run(dataset, seed)

    opt_cfg = dict(OPTIMISER_CONFIG["pso"])
    opt = OPTIMISERS["pso"](
        model=model,
        loss_fn=rmse,
        device=DEVICE,
        **opt_cfg,
    )

    x0 = opt._flatten_params()
    dim = x0.numel()

    pos = [x0 + 0.1 * torch.randn(dim) for _ in range(opt.pop_size)]
    vel = [torch.zeros(dim) for _ in range(opt.pop_size)]

    pbest = [p.clone() for p in pos]
    pbest_val = [float("inf")] * opt.pop_size

    opt._load_params(x0)
    gbest = x0.clone()
    gbest_val = opt.evaluate(Xtr, ytr)
    best_state = opt.snapshot()

    rows: List[Dict[str, float]] = []

    for it in range(1, opt.iters + 1):
        for i in range(opt.pop_size):
            opt._load_params(pos[i])
            val = opt.evaluate(Xtr, ytr)

            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest[i] = pos[i].clone()

            if val < gbest_val:
                gbest_val = val
                gbest = pos[i].clone()
                best_state = opt.snapshot()

        opt.restore(best_state)
        test_rmse = opt.evaluate(Xte, yte)

        rows.append(
            {
                "step": it,
                "step_type": "iter",
                "train_rmse": float(gbest_val),
                "test_rmse": float(test_rmse),
            }
        )

        for i in range(opt.pop_size):
            r1 = torch.rand(dim)
            r2 = torch.rand(dim)
            vel[i] = (
                opt.w * vel[i]
                + opt.c1 * r1 * (pbest[i] - pos[i])
                + opt.c2 * r2 * (gbest - pos[i])
            )
            pos[i] = pos[i] + vel[i]

    opt.restore(best_state)
    return rows


def save_curve(dataset: str, optimiser: str, seed: int, rows: List[Dict[str, float]]):
    out_dir = ROOT / "experiments" / "chenchao" / "results" / "curves" / dataset / optimiser
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"seed{seed}.csv"

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "optimiser",
                "seed",
                "step",
                "step_type",
                "train_rmse",
                "test_rmse",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "dataset": dataset,
                    "optimiser": optimiser,
                    "seed": seed,
                    **r,
                }
            )

    print(f"[curve] {out_csv}")


def parse_int_csv(raw: str) -> List[int]:
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def parse_args() -> Tuple[List[str], List[str], List[int]]:
    parser = argparse.ArgumentParser(description="Run error curves for selected datasets/optimisers/seeds.")
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help=f"Comma-separated datasets. Available: {','.join(sorted(DATASET_CONFIG.keys()))}",
    )
    parser.add_argument(
        "--optimisers",
        default=",".join(OPTIMISERS_TO_RUN),
        help="Comma-separated optimisers (supported in this script: adam,pso)",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in SEEDS),
        help="Comma-separated integer seeds.",
    )

    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    optimisers = [o.strip() for o in args.optimisers.split(",") if o.strip()]
    seeds = parse_int_csv(args.seeds)

    for d in datasets:
        if d not in DATASET_CONFIG:
            raise ValueError(f"Unknown dataset: {d}")

    for o in optimisers:
        if o not in {"adam", "pso"}:
            raise ValueError(f"Unsupported optimiser for curve runner: {o}")

    if not seeds:
        raise ValueError("No seeds parsed from --seeds")

    return datasets, optimisers, seeds


def main():
    datasets, optimisers, seeds = parse_args()

    print("=" * 70)
    print("Error curve runner")
    print("datasets :", datasets)
    print("optimisers:", optimisers)
    print("seeds    :", seeds)
    print("=" * 70)

    for dataset in datasets:
        for optimiser in optimisers:
            for seed in seeds:
                print(f"\n[run] dataset={dataset} optimiser={optimiser} seed={seed}")

                if optimiser == "adam":
                    rows = run_adam_curve(dataset, seed)
                elif optimiser == "pso":
                    rows = run_pso_curve(dataset, seed)
                else:
                    raise ValueError(f"Unsupported optimiser: {optimiser}")

                save_curve(dataset, optimiser, seed, rows)

    print("\n[DONE] error curves saved.")


if __name__ == "__main__":
    main()
