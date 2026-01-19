import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = os.path.join("data", "datasets")



UCI_DATASETS = [
    "Abalone",
    "Airfoil",
    "AutoMPG",
    "Concrete",
    "Cool",
    "Energy",
    "QSAR",
    "Treasury",
    "Yacht",
]

KEEL_DATASETS = [
    "Baseball",
    "Ankara",
    "Izmir",
    "Categorical",
    "Computer",
    "Electrical",
    "Laser",
    "Prices",
    "Heat",
    "Wine",
]

SUBSET_DATASETS = [
    "AutoMPG6"
]


ALL_DATASETS = UCI_DATASETS + KEEL_DATASETS + SUBSET_DATASETS


# =======================
# Utilities
# =======================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean datasets

    """
    df = df.replace(["?", " ", ""], np.nan)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how='all')
    df = df.fillna(df.mean())

    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df)

    return df


def load_dataset(name: str, test_size: float = 0.2, random_state: int = 0):

    csv_path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path)
    df = clean_dataframe(df)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def get_all_datasets():
    """Return list of all cleaned dataset names"""
    return ALL_DATASETS


def print_dataset_shapes():
    """Print datasets"""
    print("\n Dataset Shapes:")
    for name in ALL_DATASETS:
        path = os.path.join(DATA_DIR, f"{name}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"{name:12s} → {df.shape}")
        else:
            print(f"{name:12s} → missing")


if __name__ == "__main__":
    print_dataset_shapes()
