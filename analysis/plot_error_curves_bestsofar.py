from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CURVE_ROOT = ROOT / "experiments" / "chenchao" / "results" / "curves"
OUT_DIR = ROOT / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["airfoil", "laser", "concrete"]
OPTIMISERS = ["adam", "pso"]
SEEDS = [0, 1, 2]


def load_curve(dataset: str, optimiser: str, seed: int) -> Tuple[List[int], List[float], List[float]]:
    path = CURVE_ROOT / dataset / optimiser / f"seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing curve CSV: {path}")

    steps: List[int] = []
    train: List[float] = []
    test: List[float] = []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            train.append(float(row["train_rmse"]))
            test.append(float(row["test_rmse"]))

    return steps, train, test


def best_so_far(train: List[float], test: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    best_train = float("inf")
    best_test = float("inf")
    out_train = []
    out_test = []

    for tr, te in zip(train, test):
        if tr < best_train:
            best_train = tr
            best_test = te
        out_train.append(best_train)
        out_test.append(best_test)

    return np.asarray(out_train), np.asarray(out_test)


def aggregate(dataset: str, optimiser: str, seeds: List[int]):
    all_train = []
    all_test = []
    steps_ref = None

    for seed in seeds:
        steps, train, test = load_curve(dataset, optimiser, seed)
        btrain, btest = best_so_far(train, test)

        if steps_ref is None:
            steps_ref = steps
        elif steps != steps_ref:
            raise ValueError(f"Step mismatch for {dataset}/{optimiser}/seed{seed}")

        all_train.append(btrain)
        all_test.append(btest)

    all_train = np.stack(all_train, axis=0)
    all_test = np.stack(all_test, axis=0)

    mean_train = all_train.mean(axis=0)
    std_train = all_train.std(axis=0)

    mean_test = all_test.mean(axis=0)
    std_test = all_test.std(axis=0)

    return np.asarray(steps_ref), mean_train, std_train, mean_test, std_test


def parse_int_csv(raw: str) -> List[int]:
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot best-so-far error curves from per-seed CSV files.")
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated datasets.",
    )
    parser.add_argument(
        "--optimisers",
        default=",".join(OPTIMISERS),
        help="Comma-separated optimisers.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in SEEDS),
        help="Comma-separated seeds.",
    )
    parser.add_argument(
        "--out-stem",
        default="error_curves_bestsofar",
        help="Output file stem under analysis/ (without extension).",
    )
    return parser.parse_args()


def plot(datasets: List[str], optimisers: List[str], seeds: List[int], out_stem: str):
    n_rows = len(datasets)
    n_cols = len(optimisers)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 3.2 * n_rows),
        sharex=False,
        sharey=False,
    )

    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for r, dataset in enumerate(datasets):
        for c, optimiser in enumerate(optimisers):
            ax = axes[r, c]

            steps, m_tr, s_tr, m_te, s_te = aggregate(dataset, optimiser, seeds)

            ax.plot(steps, m_tr, color="#1f77b4", label="Train (best-so-far)")
            ax.fill_between(steps, m_tr - s_tr, m_tr + s_tr, color="#1f77b4", alpha=0.15)

            ax.plot(steps, m_te, color="#d62728", linestyle="--", label="Test (at best-train)")
            ax.fill_between(steps, m_te - s_te, m_te + s_te, color="#d62728", alpha=0.15)

            ax.set_title(f"{dataset} / {optimiser}")
            ax.set_xlabel("epoch" if optimiser == "adam" else "iteration")
            ax.set_ylabel("RMSE")
            ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_png = OUT_DIR / f"{out_stem}.png"
    out_pdf = OUT_DIR / f"{out_stem}.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}")
    print(f"saved: {out_pdf}")


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    optimisers = [o.strip() for o in args.optimisers.split(",") if o.strip()]
    seeds = parse_int_csv(args.seeds)

    if not datasets:
        raise ValueError("No datasets provided.")
    if not optimisers:
        raise ValueError("No optimisers provided.")
    if not seeds:
        raise ValueError("No seeds provided.")

    plot(datasets=datasets, optimisers=optimisers, seeds=seeds, out_stem=args.out_stem)


if __name__ == "__main__":
    main()
