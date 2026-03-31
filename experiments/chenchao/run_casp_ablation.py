from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.experiment_config import (
    DEVICE,
    INIT_SOURCE,
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
DEFAULT_DATASETS = ["laser"]
DEFAULT_OPTIMISERS = ["adam"]
DEFAULT_SEEDS = list(range(10))
CASP_MODES = ["single", "adapted", "free"]


def rmse(y, yhat):
    return torch.sqrt(torch.mean((y - yhat) ** 2))


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_seeds(s: str) -> List[int]:
    out: List[int] = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            out.extend(list(range(int(a), int(b) + 1)))
        else:
            out.append(int(token))
    return sorted(set(out))


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


def run_single(dataset: str, seed: int, optimiser: str, casp_mode: str) -> Tuple[float, float]:
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
        casp_mode=casp_mode,
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


def main():
    parser = argparse.ArgumentParser(description="CASP mode ablation")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--optimisers", default=",".join(DEFAULT_OPTIMISERS))
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    datasets = parse_list(args.datasets)
    optimisers = parse_list(args.optimisers)
    seeds = parse_seeds(args.seeds)

    if args.out:
        out_csv = Path(args.out)
    else:
        ds_tag = "-".join(datasets)
        opt_tag = "-".join(optimisers)
        seed_tag = f"{seeds[0]}_{seeds[-1]}" if len(seeds) > 1 else str(seeds[0])
        out_csv = ROOT / "experiments" / "chenchao" / "results" / f"casp_ablation_{ds_tag}_{opt_tag}_seed{seed_tag}.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CASP ablation")
    print("datasets :", datasets)
    print("optimisers:", optimisers)
    print("seeds    :", seeds)
    print("casp_modes:", CASP_MODES)
    print("output   :", out_csv)
    print("=" * 70)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "optimiser", "seed", "casp_mode", "train_rmse", "test_rmse"],
        )
        writer.writeheader()

        total = len(datasets) * len(optimisers) * len(seeds) * len(CASP_MODES)
        k = 1

        for dataset in datasets:
            for optimiser in optimisers:
                for seed in seeds:
                    for casp_mode in CASP_MODES:
                        print(f"\n[{k}/{total}] dataset={dataset} opt={optimiser} seed={seed} casp={casp_mode}")
                        k += 1
                        tr, te = run_single(dataset, seed, optimiser, casp_mode)
                        writer.writerow({
                            "dataset": dataset,
                            "optimiser": optimiser,
                            "seed": seed,
                            "casp_mode": casp_mode,
                            "train_rmse": tr,
                            "test_rmse": te,
                        })
                        f.flush()

    print("\n[DONE] CASP ablation finished.")


if __name__ == "__main__":
    main()
