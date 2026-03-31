from __future__ import annotations

import argparse
import csv
import os
import sys
import time
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
    CASP_MODE,
    DATASET_CONFIG,
    FOLD,
)
from src.FuzzyInferenceSystem import FuzzyInferenceSystem
from src.trainable_fis import TrainableFIS
from src.fis_casps import decode_trapmf


def load_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("@"):
                continue
            rows.append([float(x) for x in s.replace(",", " ").split()])
    arr = np.asarray(rows, dtype=np.float32)
    return arr[:, :-1], arr[:, -1]


def load_theta(path: str) -> List[torch.Tensor]:
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


def build_model(dataset: str, seed: int):
    cfg = DATASET_CONFIG[dataset]
    base = ROOT / "experiments" / "chenchao" / "data"

    tr_p, _, th_p, ru_p = resolve_dataset_paths(
        base, dataset, FOLD, seed, INIT_SOURCE, cfg["use_pca"]
    )

    Xtr_np, _ = load_xy(str(tr_p))

    if cfg.get("input_range_normalised", False):
        lo = Xtr_np.min(axis=0)
        hi = Xtr_np.max(axis=0)
        span = hi - lo
        span[span == 0.0] = 1.0
        Xtr_np = (Xtr_np - lo) / span

    Xtr = torch.tensor(Xtr_np, device=DEVICE)

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

    return model, Xtr


def main():
    parser = argparse.ArgumentParser(description="Benchmark Python FIS inference speed")
    parser.add_argument("--dataset", default="laser")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    torch.set_num_threads(1)
    np.random.seed(args.seed)

    model, Xtr = build_model(args.dataset, args.seed)

    n = min(args.samples, Xtr.shape[0])
    idx = np.random.choice(Xtr.shape[0], size=n, replace=False)
    X = Xtr[idx]

    model.eval()
    totals = []
    means = []

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(X)

        for trial in range(1, args.trials + 1):
            t0 = time.perf_counter()
            for _ in range(args.repeats):
                _ = model(X)
            t1 = time.perf_counter()

            total = t1 - t0
            mean_ms = (total / args.repeats) * 1000.0

            totals.append(total)
            means.append(mean_ms)

            print(f"[trial {trial}] total_s={total:.6f} mean_ms={mean_ms:.6f}")

    total_mean = float(np.mean(totals))
    mean_ms_mean = float(np.mean(means))
    mean_ms_median = float(np.median(means))
    mean_ms_std = float(np.std(means, ddof=1)) if len(means) > 1 else float("nan")

    print("=" * 70)
    print("Python FIS benchmark")
    print("dataset:", args.dataset)
    print("seed   :", args.seed)
    print("samples:", n)
    print("repeats:", args.repeats)
    print("trials :", args.trials)
    print("mean_total_s:", total_mean)
    print("mean_ms_mean:", mean_ms_mean)
    print("mean_ms_median:", mean_ms_median)
    print("mean_ms_std:", mean_ms_std)
    print("=" * 70)

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = ROOT / "experiments" / "benchmarks" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "benchmark_fis_python_trials.csv"

    write_header = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "language",
                "dataset",
                "seed",
                "samples",
                "repeats",
                "trial",
                "total_seconds",
                "mean_ms",
            ],
        )
        if write_header:
            w.writeheader()
        for i, (total, mean_ms) in enumerate(zip(totals, means), start=1):
            w.writerow(
                {
                    "language": "python",
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "samples": n,
                    "repeats": args.repeats,
                    "trial": i,
                    "total_seconds": total,
                    "mean_ms": mean_ms,
                }
            )

    print("saved:", out_path)


if __name__ == "__main__":
    main()
