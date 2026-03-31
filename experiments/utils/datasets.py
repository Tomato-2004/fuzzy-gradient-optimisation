# experiments/utils/datasets.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

DATA_DIR = os.path.join("data", "datasets")

UCI_DATASETS = [
    "Abalone", "Airfoil", "AutoMPG", "Concrete",
    "Cool", "Energy", "QSAR", "Treasury", "Yacht",
]

KEEL_DATASETS = [
    "Baseball", "Ankara", "Izmir", "Categorical",
    "Computer", "Electrical", "Laser", "Prices",
    "Heat", "Wine",
]

SUBSET_DATASETS = ["AutoMPG6"]
ALL_DATASETS = UCI_DATASETS + KEEL_DATASETS + SUBSET_DATASETS


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(["?", " ", ""], np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.fillna(df.mean())
    return df


def load_dataset(
    name: str,
    test_size: float = 0.2,
    random_state: int = 0,
    scale_y: bool = False,
    return_scaler: bool = False,
):
    csv_path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path)
    df = clean_dataframe(df)

    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.reshape(-1, 1).astype(float)

    # ---------------- split FIRST (avoid leakage) ----------------
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    meta = {}

    # ---------------- PCA for Concrete (Chen Chao style) ----------------
    # Fit PCA on train only; transform both
    if name.lower() == "concrete":
        pca = PCA(n_components=5, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        meta["pca"] = pca
        meta["pca_n_components"] = 5

    # ---------------- scale X to [0,1] (train-fit) ----------------
    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    meta["scaler_X"] = scaler_X

    # ---------------- optionally scale y ----------------
    if scale_y:
        scaler_y = MinMaxScaler()
        y_train = scaler_y.fit_transform(y_train_raw).reshape(-1)
        y_test = scaler_y.transform(y_test_raw).reshape(-1)
        meta["scaler_y"] = scaler_y
        meta["y_min"] = float(scaler_y.data_min_[0])
        meta["y_max"] = float(scaler_y.data_max_[0])
    else:
        y_train = y_train_raw.reshape(-1)
        y_test = y_test_raw.reshape(-1)

    if return_scaler:
        return X_train, X_test, y_train, y_test, meta
    return X_train, X_test, y_train, y_test


def get_all_datasets():
    return ALL_DATASETS
