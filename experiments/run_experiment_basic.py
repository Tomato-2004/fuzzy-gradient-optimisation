import os
import sys
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from experiments.utils.datasets import UCI_DATASETS
from experiments.train_adam import train_one_dataset
from experiments.train_pso import train_one_dataset_pso


def main():
    """
    在所有 UCI 数据集上分别运行：
        - Adam 优化
        - PSO 优化（非梯度）
    输出结果到 experiments/results_compare_adam_pso.csv
    """

    results = []

    for name in UCI_DATASETS:

        print("\n======================================")
        print(f"Running dataset: {name}")
        print("======================================\n")

        # ===== 运行 Adam =====
        res_adam = train_one_dataset(
            dataset_name=name,
            n_mfs_per_input=3,
            n_mfs_output=3,
            n_rules=10,
            num_epochs=100,
            lr=1e-3,
        )

        # ===== 运行 PSO =====
        res_pso = train_one_dataset_pso(
            dataset_name=name,
            n_mfs_per_input=3,
            n_mfs_output=3,
            n_rules=10,
            num_epochs=30,      # PSO 通常更慢，可以设少一些
            swarm_size=20,
        )

        results.append({
            "dataset": name,
            "adam_train_mse": res_adam["train_mse"],
            "adam_test_mse":  res_adam["test_mse"],
            "pso_train_mse":  res_pso["train_mse"],
            "pso_test_mse":   res_pso["test_mse"],
        })

    # === 保存到 CSV ===
    out_path = os.path.join("experiments", "results_compare_adam_pso.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "adam_train_mse", "adam_test_mse",
            "pso_train_mse", "pso_test_mse"
        ])
        for r in results:
            writer.writerow([
                r["dataset"],
                r["adam_train_mse"], r["adam_test_mse"],
                r["pso_train_mse"], r["pso_test_mse"]
            ])

    print(f"\nAll datasets finished. Results saved to {out_path}")


if __name__ == "__main__":
    main()
