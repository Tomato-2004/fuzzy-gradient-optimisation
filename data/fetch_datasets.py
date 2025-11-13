"""
fetch_datasets.py
下载论文使用的 13 个 UCI 回归数据集（排除 KEEL）
保存到 data/datasets/
"""

import os
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler

SAVE_DIR = os.path.join("data", "datasets")
os.makedirs(SAVE_DIR, exist_ok=True)

UCI_DATASETS = [
    "Airfoil_self_noise",
    "Concrete_Compressive_Strength",
    "autoMpg",
    "Yacht_hydrodynamics",
    "qsar_aquatic_toxicity",
    "Abalone",
    "Computer_hardware",
    "Energy_efficiency",
    "wine-quality-red",
    "wine-quality-white",
    "Baseball",
    "Treasury",
    "Laser",
]

print("\n=== 开始下载 13 个 UCI 回归数据集 ===")

for name in UCI_DATASETS:
    csv_path = os.path.join(SAVE_DIR, f"{name}.csv")

    if os.path.exists(csv_path):
        print(f"✅ Already exists: {name}")
        continue

    print(f"🔍 Fetching from OpenML: {name}")
    try:
        ds = fetch_openml(name, as_frame=True, parser="pandas")
        X, y = ds.data, ds.target

        # 只保留数值列
        df = pd.concat([X, y], axis=1)
        df = df.apply(pd.to_numeric, errors="coerce").dropna()

        # 归一化
        scaler = MinMaxScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])

        df.to_csv(csv_path, index=False)
        print(f"✅ Saved: {csv_path}")
    except Exception as e:
        print(f"⚠️  Skipped {name}: {e}")

print("\n🎯 所有 13 个 UCI 数据集已准备完毕！")
