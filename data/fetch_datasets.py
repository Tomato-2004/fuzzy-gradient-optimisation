import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

SRC_DIR = "data/datasets"
DST_DIR = "data/datasets"  
os.makedirs(DST_DIR, exist_ok=True)

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

    df = df.apply(pd.to_numeric, errors='coerce')

    df = df.fillna(df.mean())

    # Normalize
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df)

    if df.columns[-1].lower() not in ["target", "class", "output"]:
        pass

    # Save
    df.to_csv(src_path, index=False)
    print(f"✔ Saved {name}.csv (shape={df.shape})")


print("=== Processing existing clean datasets ===")

for d in DATASETS:
    clean_dataset(d)

print("\n🎯 All DONE! Your data is ready for machine learning 🚀")
