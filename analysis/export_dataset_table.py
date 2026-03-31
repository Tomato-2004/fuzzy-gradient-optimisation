import os
import pandas as pd

DATASET_DIR = "data/datasets"

records = []

for fname in sorted(os.listdir(DATASET_DIR)):
    if not fname.endswith(".csv"):
        continue

    path = os.path.join(DATASET_DIR, fname)
    df = pd.read_csv(path)

    # 约定：最后一列是 target，其余是 features
    n_samples = df.shape[0]
    n_features = df.shape[1] - 1

    records.append({
        "Dataset": os.path.splitext(fname)[0],
        "N": n_samples,
        "d": n_features
    })

dataset_table = pd.DataFrame(records).sort_values("Dataset")
dataset_table.to_csv("analysis/table_datasets.csv", index=False)

print(dataset_table)
