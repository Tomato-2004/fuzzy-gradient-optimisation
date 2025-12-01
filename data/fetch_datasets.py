"""
Download ALL 20 datasets used in the Mathematics CASP paper
and save them into data/datasets/.

Includes:
- 13 datasets from your previous list
- Additional 7 missing datasets (mostly OpenML & UCI originals)
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler

SAVE_DIR = os.path.join("data", "datasets")
os.makedirs(SAVE_DIR, exist_ok=True)

# === Full 20 datasets (CASP paper) ===
DATASETS_20 = {

    # Already downloaded
    "Abalone": "Abalone",
    "Airfoil_self_noise": "Airfoil_self_noise",
    "AutoMPG": "autoMpg",
    "Baseball": "Baseball",
    "Computer_hardware": "Computer_hardware",
    "Concrete_Compressive_Strength": "Concrete_Compressive_Strength",
    "Energy_efficiency": "Energy_efficiency",
    "Laser": "Laser",
    "QSAR_aquatic_toxicity": "qsar_aquatic_toxicity",
    "Treasury": "Treasury",
    "wine-quality-red": "wine-quality-red",
    "wine-quality-white": "wine-quality-white",
    "Yacht_Hydrodynamics": "Yacht_hydrodynamics",

    # Missing 7 datasets below:
    "Concrete_Slump": "slump_test",
    "Forest_Fires": "forestfires",
    "Housing": "boston",                   # Boston Housing (removed from sklearn)
    "Bike_Sharing": "Bike_Sharing_Demand",
}


def clean_df(df):
    """Convert to float + fill NaN + remove non-numeric columns."""
    df = df.replace(["?", ""], np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.fillna(df.mean())
    return df


def save_dataset(name, df):
    df = clean_df(df)
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df)
    path = os.path.join(SAVE_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"✔ Saved: {path}")


print("\n=== Downloading ALL 20 CASP datasets ===")

for name, openml_id in DATASETS_20.items():

    save_path = os.path.join(SAVE_DIR, f"{name}.csv")
    if os.path.exists(save_path):
        print(f"✔ Already exists: {name}")
        continue

    print(f"⬇ Fetching: {name}  (OpenML key: {openml_id})")

    try:
        ds = fetch_openml(openml_id, as_frame=True, parser="pandas")
        X, y = ds.data, ds.target
        df = pd.concat([X, y], axis=1)
        save_dataset(name, df)
        continue

    except Exception as e:
        print(f"⚠ OpenML not available for {name}: {e}")

    # Try UCI manual URLs (special cases)
    if name == "Housing":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
        columns = [
            'CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'
        ]
        df = pd.read_csv(url, delim_whitespace=True, header=None, names=columns)
        save_dataset(name, df)
        continue

    if name == "Concrete_Slump":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data"
        df = pd.read_csv(url)
        save_dataset(name, df)
        continue

    if name == "Forest_Fires":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
        df = pd.read_csv(url)
        save_dataset(name, df)
        continue

    print(f"❌ Could not fetch {name}, manual download required.")


print("\n🎉 All available datasets processed!")
