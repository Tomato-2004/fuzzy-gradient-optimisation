# experiments/run_experiment_basic.py
import os
import sys
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from experiments.utils.datasets import UCI_DATASETS
from experiments.train_adam import train_one_dataset


def main():
    results = []

    for name in UCI_DATASETS:
        res = train_one_dataset(
            name,
            n_mfs_per_input=3,
            n_mfs_output=3,
            n_rules=10,
            num_epochs=100,
            lr=1e-3,
        )
        results.append(res)

    # 保存一个简单的结果表，之后可以对照 CASP 论文最后的表格形式
    out_path = os.path.join("experiments", "results_adam_basic.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "train_mse", "test_mse"])
        for r in results:
            writer.writerow([r["dataset"], r["train_mse"], r["test_mse"]])

    print(f"\nAll datasets finished. Results saved to {out_path}")


if __name__ == "__main__":
    main()
