# experiments/utils/datasets.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join("data", "datasets")

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


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    通用数据清洗：
        - 把 '?' 和空字符串当作缺失值
        - 所有列转换为 float
        - 用列均值填充 NaN
    """

    # 把 "?" 和 "" 替换为 NaN
    df = df.replace(["?", " ", ""], np.nan)

    # 尝试转换为 float（非数值列会自动变成 NaN）
    df = df.apply(pd.to_numeric, errors='coerce')

    # 用每列均值填充 NaN（AutoMPG、Baseball 等数据集需要）
    df = df.fillna(df.mean())

    return df


def load_dataset(name: str, test_size: float = 0.2, random_state: int = 0):
    """
    从 data/datasets/{name}.csv 读取，默认最后一列为 y。
    返回：X_train, X_test, y_train, y_test（numpy）
    """

    csv_path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    # 读取 CSV
    df = pd.read_csv(csv_path)

    # 清洗数据（关键步骤）
    df = clean_dataframe(df)

    # 划分 X / y
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(float)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
