import pandas as pd
import numpy as np

# =========================
# Load results
# =========================

df = pd.read_csv("experiments/results_final_validation_all.csv")

# Sanity check
print(df.head())
print(df.groupby("optimiser").size())

# =========================
# 1. Overall performance (paper main table)
# =========================

overall = (
    df.groupby("optimiser")
      .agg(
          mean_mse=("test_mse", "mean"),
          std_mse=("test_mse", "std"),
          mean_time=("train_time", "mean"),
          std_time=("train_time", "std")
      )
      .reset_index()
      .sort_values("mean_mse")
)

print("\n=== Overall performance ===")
print(overall)

overall.to_csv("analysis/table_overall_results.csv", index=False)

# =========================
# 2. Per-dataset median (for win-rate)
# =========================

per_dataset = (
    df.groupby(["dataset", "optimiser"])
      .agg(
          median_mse=("test_mse", "median"),
          mean_time=("train_time", "mean")
      )
      .reset_index()
)

per_dataset.to_csv("analysis/table_per_dataset_summary.csv", index=False)

# =========================
# 3. Win-rate analysis
# =========================

wins = (
    per_dataset.loc[
        per_dataset.groupby("dataset")["median_mse"].idxmin()
    ]
    .groupby("optimiser")
    .size()
    .reset_index(name="num_wins")
    .sort_values("num_wins", ascending=False)
)

print("\n=== Win counts ===")
print(wins)

wins.to_csv("analysis/table_win_counts.csv", index=False)

# =========================
# 4. Ranking analysis (average rank)
# =========================

per_dataset["rank"] = (
    per_dataset.groupby("dataset")["median_mse"]
    .rank(method="average")
)

avg_rank = (
    per_dataset.groupby("optimiser")["rank"]
    .mean()
    .reset_index(name="avg_rank")
    .sort_values("avg_rank")
)

print("\n=== Average ranks ===")
print(avg_rank)

avg_rank.to_csv("analysis/table_average_ranks.csv", index=False)

# =========================
# 5. Time–accuracy trade-off (scatter source)
# =========================

tradeoff = (
    df.groupby("optimiser")
      .agg(
          mean_mse=("test_mse", "mean"),
          mean_time=("train_time", "mean")
      )
      .reset_index()
)

tradeoff.to_csv("analysis/table_time_accuracy.csv", index=False)

print("\nSaved all analysis tables to ./analysis/")
