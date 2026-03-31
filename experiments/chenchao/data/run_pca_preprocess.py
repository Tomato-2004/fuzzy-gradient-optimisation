from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from io import StringIO

# ======================================================
# CONFIG
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PCA_DATASETS = {
    "abalone": 5,
    # "concrete": 5,
    # "ankara": 5,
    # "izmir": 5,
    # "baseball": 5,
    # "treasury": 5,
    # "wine": 5,
}

TEST_RATIO = 0.2


# ======================================================
# Utils
# ======================================================
def load_chenchao_dat(path: str) -> pd.DataFrame:
    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("@") or line.startswith("%"):
                continue
            lines.append(line)

    sep = "," if "," in lines[0] else r"\s+"
    return pd.read_csv(StringIO("\n".join(lines)), header=None, sep=sep)


# ======================================================
# PCA routine (teacher-correct)
# ======================================================
def process_dataset(name: str, n_comp: int):
    print(f"\n=== PCA processing {name} ===")

    in_path = os.path.join(BASE_DIR, name, f"{name}.dat")
    if not os.path.exists(in_path):
        print(f"[SKIP] {in_path} not found")
        return

    data = load_chenchao_dat(in_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].reset_index(drop=True)

    # ---- PCA (FIT ON FULL DATA)
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)

    data_pca = pd.concat(
        [pd.DataFrame(X_pca), y],
        axis=1
    )

    # ---- output dirs
    pca_dir = os.path.join(BASE_DIR, name, f"{name}-pca")
    fold_dir = os.path.join(pca_dir, "5-fold")
    os.makedirs(fold_dir, exist_ok=True)

    # ---- 1) dataset-pca.dat
    pca_dat_path = os.path.join(pca_dir, f"{name}-pca.dat")
    data_pca.to_csv(pca_dat_path, index=False, header=False)

    # ---- 2) 5-1tra.dat  (EXACT COPY)
    tra_path = os.path.join(fold_dir, f"{name}-pca-5-1tra.dat")
    data_pca.to_csv(tra_path, index=False, header=False)

    # ---- 3) 5-1tst.dat  (external-style test)
    n = len(data_pca)
    split = int((1.0 - TEST_RATIO) * n)
    tst = data_pca.iloc[split:].copy()

    tst_path = os.path.join(fold_dir, f"{name}-pca-5-1tst.dat")
    tst.to_csv(tst_path, index=False, header=False)

    print("Explained variance:", pca.explained_variance_ratio_)
    print(f"[DONE] {name}")


# ======================================================
# Entry
# ======================================================
if __name__ == "__main__":
    for ds, k in PCA_DATASETS.items():
        process_dataset(ds, k)
