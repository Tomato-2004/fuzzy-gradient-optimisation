import pandas as pd
import os


def merge_experiment_results():
    """
    Merge full grid results with gradient baseline results.
    """

    grid_path = os.path.join("experiments", "results_full_grid.csv")
    grad_path = os.path.join("experiments", "results_gradient_baselines.csv")
    out_path = os.path.join("experiments", "results_full_grid_extended.csv")

    print("Loading CSV files...")
    df_grid = pd.read_csv(grid_path)
    df_grad = pd.read_csv(grad_path)

    print(f"  Full grid rows      : {len(df_grid)}")
    print(f"  Gradient baseline rows: {len(df_grad)}")

    if list(df_grid.columns) != list(df_grad.columns):
        raise ValueError(
            "CSV column mismatch:\n"
            f"Grid columns: {df_grid.columns.tolist()}\n"
            f"Grad columns: {df_grad.columns.tolist()}"
        )

    print("Merging results...")
    df_all = pd.concat([df_grid, df_grad], ignore_index=True)

    print(f"Total rows after merge: {len(df_all)}")

    print(f"Saving merged CSV to: {out_path}")
    df_all.to_csv(out_path, index=False)

    print("Merge completed successfully.")


if __name__ == "__main__":
    merge_experiment_results()
