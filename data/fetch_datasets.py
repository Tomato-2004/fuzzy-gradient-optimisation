import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

SRC_DIR = "data/datasets"
DST_DIR = "data/datasets"  # 直接覆盖原文件，也可以改成新文件夹
os.makedirs(DST_DIR, exist_ok=True)

# Current datasets (17 total)
DATASETS = [
    "Abalone",
    "Airfoil",
    "Ankara",
    "AutoMPG",
    "AutoMPG6",
    "Baseball",
    "Categorical",
    "Computer",
    "Concrete",
    "Cool",
    "Electrical",
    "Energy",
    "Heat",
    "Izmir",
    "Laser",
    "Prices",
    "QSAR",
    "Treasury",
    "Wine",
    "Yacht"
]


def clean_dataset(name):
    src_path = os.path.join(SRC_DIR, f"{name}.csv")

    if not os.path.exists(src_path):
        print(f"⚠ Missing: {name}.csv")
        return

    print(f"\n🧹 Cleaning {name}.csv")

    df = pd.read_csv(src_path)

    # Ensure numeric conversion
    df = df.apply(pd.to_numeric, errors='coerce')

    # Fill NaN with column mean
    df = df.fillna(df.mean())

    # Normalize
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df)

    # Move last column to be target (if not already last)
    if df.columns[-1].lower() not in ["target", "class", "output"]:
        # Assume last column currently is the target — do nothing
        pass

    # Save
    df.to_csv(src_path, index=False)
    print(f"✔ Saved {name}.csv (shape={df.shape})")


print("=== Processing existing clean datasets ===")

for d in DATASETS:
    clean_dataset(d)

print("\n🎯 All DONE! Your data is ready for machine learning 🚀")
