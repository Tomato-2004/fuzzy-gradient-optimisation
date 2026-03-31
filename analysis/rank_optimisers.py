from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGE3 = ROOT / "experiments" / "chenchao" / "results" / "stage3_seed0_9_all_optimisers.csv"
OUT_DIR = ROOT / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_rows(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rankdata(values: List[float]) -> List[float]:
    # average ranks for ties, 1-based
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks.tolist()


def main():
    rows = load_rows(STAGE3)
    datasets = sorted({r["dataset"] for r in rows})
    optimisers = sorted({r["optimiser"] for r in rows})

    # mean test RMSE per dataset/optimiser over seeds
    mean_by = {}
    for ds in datasets:
        for opt in optimisers:
            vals = [float(r["test_rmse"]) for r in rows if r["dataset"] == ds and r["optimiser"] == opt]
            mean_by[(ds, opt)] = float(np.mean(vals)) if vals else float("nan")

    # dataset-level ranks
    rank_rows = []
    for ds in datasets:
        means = [mean_by[(ds, opt)] for opt in optimisers]
        ranks = rankdata(means)
        for opt, mean, rank in zip(optimisers, means, ranks):
            rank_rows.append({
                "dataset": ds,
                "optimiser": opt,
                "mean_test_rmse": mean,
                "rank": rank,
            })

    out_rank = OUT_DIR / "optimizer_ranks_by_dataset.csv"
    with out_rank.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "optimiser", "mean_test_rmse", "rank"])
        w.writeheader()
        w.writerows(rank_rows)

    # average rank across datasets (scale-free summary)
    avg_rank_rows = []
    for opt in optimisers:
        ranks = [r["rank"] for r in rank_rows if r["optimiser"] == opt]
        avg_rank_rows.append({
            "optimiser": opt,
            "avg_rank": float(np.mean(ranks)),
            "std_rank": float(np.std(ranks, ddof=1)) if len(ranks) > 1 else float("nan"),
        })

    avg_rank_rows = sorted(avg_rank_rows, key=lambda x: x["avg_rank"])

    out_avg = OUT_DIR / "optimizer_avg_rank.csv"
    with out_avg.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["optimiser", "avg_rank", "std_rank"])
        w.writeheader()
        w.writerows(avg_rank_rows)

    print(f"saved: {out_rank}")
    print(f"saved: {out_avg}")


if __name__ == "__main__":
    main()
