# experiments/utils/datasets.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join("data", "datasets")

# 这里用你截图里的文件名
UCI_DATASETS = [
    "Abalone",
    "Airfoil_self_noise",
    "AutoMPG",
    "Baseball",
    "Computer_hardware",
    "Concrete_Compressive_Strength",
    "Energy_efficiency",
    "Laser",
    "QSAR_aquatic_toxicity",
    "Treasury",
    "wine-quality-red",
    "wine-quality-white",
    "Yacht_Hydrodynamics",
]


def load_dataset(name: str, test_size: float = 0.2, random_state: int = 0):
    """
    从 data/datasets/{name}.csv 读取，默认最后一列为 y。
    返回：X_train, X_test, y_train, y_test（numpy）
    """
    csv_path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path)

    # 假定所有列都是数值，最后一列是目标
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
