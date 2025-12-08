import os
import pandas as pd

ROOT_DIR = "data"
SRC_DIR = ROOT_DIR                # source files are here
DST_DIR = os.path.join(ROOT_DIR, "datasets")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ============================================================
# 1. Convert Cool dataset (Excel → CSV)
# ============================================================

def convert_cool():
    src = os.path.join(SRC_DIR, "ENB2012_data.xlsx")
    dst = os.path.join(DST_DIR, "Cool.csv")

    if not os.path.exists(src):
        print(f"[Cool] ❌ Source not found: {src}")
        return

    print(f"[Cool] Converting: {src} → {dst}")
    df = pd.read_excel(src)
    df.to_csv(dst, index=False)
    print("[Cool] ✔ Done")


# ============================================================
# 2. Convert Categorical dataset (ANACALT → CSV)
# ============================================================

def convert_categorical():
    src = os.path.join(SRC_DIR, "ANACALT.dat")
    dst = os.path.join(DST_DIR, "Categorical.csv")

    if not os.path.exists(src):
        print(f"[Categorical] ❌ Source not found: {src}")
        return

    print(f"[Categorical] Converting: {src} → {dst}")

    rows = []
    with open(src, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("@"):
                continue
            rows.append(line.split(","))

    df = pd.DataFrame(rows)
    df.to_csv(dst, index=False)
    print("[Categorical] ✔ Done")


# ============================================================
# Run all conversions
# ============================================================

if __name__ == "__main__":
    print("=== Converting local datasets ===")
    ensure_dir(DST_DIR)

    convert_cool()
    convert_categorical()

    print("\nAll conversions complete.")
