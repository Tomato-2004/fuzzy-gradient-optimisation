import os
import pandas as pd
import numpy as np

# =========================
# Paths
# =========================

INPUT_CSV = "experiments/results_final_validation_all.csv"
OUT_DIR = "analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Load results (Stage III)
# =========================

df = pd.read_csv(INPUT_CSV)

# Basic sanity checks
print("=== Head ===")
print(df.head())
print("\n=== Rows per optimiser ===")
print(df.groupby("optimiser").size())

required_cols = {"dataset", "optimiser", "seed", "test_mse", "train_time"}
assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"

# =========================
# 1. Overall performance table (main paper table)
# Mean ± std across all datasets and seeds
# =========================

overall = (
    df.groupby("optimiser")
      .agg(
          mean_mse=("test_mse", "mean"),
          std_mse=("test_mse", "std"),
          mean_time=("train_time", "mean"),
          std_time=("train_time", "std"),
      )
      .reset_index()
      .sort_values("mean_mse")
)

print("\n=== Overall performance (mean ± std) ===")
print(overall)

overall.to_csv(
    os.path.join(OUT_DIR, "table_overall_results.csv"),
    index=False
)

# =========================
# 2. Per-dataset aggregation (median over seeds)
# Used for win-rate, ranking, and statistical tests
# =========================

per_dataset = (
    df.groupby(["dataset", "optimiser"])
      .agg(
          median_mse=("test_mse", "median"),
          mean_time=("train_time", "mean"),
      )
      .reset_index()
)

per_dataset.to_csv(
    os.path.join(OUT_DIR, "table_per_dataset_summary.csv"),
    index=False
)

# =========================
# 3. Win-rate analysis
# A win = lowest median test MSE on a dataset
# =========================

idx_best = per_dataset.groupby("dataset")["median_mse"].idxmin()

wins = (
    per_dataset.loc[idx_best]
      .groupby("optimiser")
      .size()
      .reset_index(name="num_wins")
      .sort_values("num_wins", ascending=False)
)

print("\n=== Win-rate (count of datasets won) ===")
print(wins)

wins.to_csv(
    os.path.join(OUT_DIR, "table_win_counts.csv"),
    index=False
)

# =========================
# 4. Ranking analysis
# Rank optimisers per dataset using median MSE
# =========================

per_dataset["rank"] = (
    per_dataset.groupby("dataset")["median_mse"]
    .rank(method="average", ascending=True)
)

avg_rank = (
    per_dataset.groupby("optimiser")["rank"]
    .mean()
    .reset_index(name="avg_rank")
    .sort_values("avg_rank")
)

print("\n=== Average ranks across datasets ===")
print(avg_rank)

avg_rank.to_csv(
    os.path.join(OUT_DIR, "table_average_ranks.csv"),
    index=False
)

# =========================
# 5. Accuracy–efficiency trade-off table
# Used for scatter plots or summary tables
# =========================

tradeoff = (
    df.groupby("optimiser")
      .agg(
          mean_mse=("test_mse", "mean"),
          mean_time=("train_time", "mean"),
      )
      .reset_index()
      .sort_values("mean_mse")
)

tradeoff.to_csv(
    os.path.join(OUT_DIR, "table_time_accuracy.csv"),
    index=False
)

print(f"\nAll analysis tables saved to ./{OUT_DIR}/")
