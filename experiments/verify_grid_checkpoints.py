# experiments/verify_grid_checkpoints.py
from __future__ import annotations

import os
import sys
import csv
import ast
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from experiments.utils.datasets import load_dataset
from src.trainable_fis import TrainableFIS


CSV_PATH = "experiments/grid_results/results_grid.csv"
OUT_CSV  = "experiments/grid_results/results_grid_verified.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Utils
# ============================================================
def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    yhat = np.asarray(yhat, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def inverse_y(y_scaled: np.ndarray, meta: dict) -> np.ndarray:
    if isinstance(meta, dict) and meta.get("scaler_y", None) is not None:
        s = meta["scaler_y"]
        return s.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).reshape(-1)

    # fallback
    if isinstance(meta, dict) and ("y_min" in meta) and ("y_max" in meta):
        y_min = float(meta["y_min"])
        y_max = float(meta["y_max"])
        return np.asarray(y_scaled).reshape(-1) * (y_max - y_min) + y_min

    return np.asarray(y_scaled).reshape(-1)


def parse_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, float):
            return x
        if isinstance(x, str) and x.strip() == "":
            return None
        return float(x)
    except Exception:
        return None


# ============================================================
# Verify one checkpoint
# ============================================================
def verify_one_ckpt(dataset: str, seed: int, ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = ckpt.get("config", {})
    point_n = 101
    opt = cfg.get("optimiser", {}).get("method", None)
    if opt is not None:
        key = f"{opt}_params"
        if key in cfg and isinstance(cfg[key], dict) and "point_n" in cfg[key]:
            point_n = int(cfg[key]["point_n"])
    if "point_n" in cfg:
        point_n = int(cfg["point_n"])

    # reload test split EXACTLY like grid
    # (same random_state=seed in loader call)
    X_trn, X_tst, y_trn_s, y_tst_s, meta = load_dataset(
        dataset,
        random_state=int(seed),
        scale_y=True,
        return_scaler=True,
    )
    _, _, _, y_tst = load_dataset(
        dataset,
        random_state=int(seed),
        scale_y=False,
        return_scaler=False,
    )

    X_tst = np.asarray(X_tst, dtype=np.float32)
    y_tst = np.asarray(y_tst, dtype=np.float32).reshape(-1)

    # reconstruct model from ckpt base_fis
    base_fis = ckpt.get("base_fis", None)
    if base_fis is None:
        raise ValueError("Checkpoint missing base_fis. Cannot rebuild model safely.")

    casp_mode = ckpt.get("casp_mode", None)
    if casp_mode is None:
        casp_mode = ckpt.get("config", {}).get("constraints", {}).get("casp_mode", "free")

    model = TrainableFIS(base_fis, casp_mode=casp_mode)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        preds_s = model(
            torch.tensor(X_tst, dtype=torch.float32, device=DEVICE),
            point_n=int(point_n),
        ).detach().cpu().numpy().reshape(-1)

    preds = inverse_y(preds_s, meta)
    score = rmse(y_tst, preds)

    return {
        "verified_rmse": float(score),
        "point_n": int(point_n),
        "ok": True,
        "err": "",
    }


# ============================================================
# Main
# ============================================================
def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    print(f"[Verify] Reading CSV: {CSV_PATH}")
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    out_rows = []
    n_ok = 0
    n_skip = 0
    n_err = 0

    for i, r in enumerate(rows, 1):
        status = r.get("status", "")
        ckpt_path = r.get("ckpt_path", "")

        # keep original row
        r2 = dict(r)

        # ignore SKIP/ERR rows
        if status != "OK":
            r2["verified_rmse"] = ""
            r2["rmse_diff"] = ""
            r2["verify_status"] = f"SKIP({status})"
            out_rows.append(r2)
            n_skip += 1
            continue

        if not ckpt_path or (not os.path.exists(ckpt_path)):
            r2["verified_rmse"] = ""
            r2["rmse_diff"] = ""
            r2["verify_status"] = "ERR(no_ckpt)"
            out_rows.append(r2)
            n_err += 1
            continue

        dataset = r["dataset"]
        seed = int(r["seed"])
        csv_rmse = parse_float(r.get("rmse_test", None))

        try:
            info = verify_one_ckpt(dataset, seed, ckpt_path)
            verified_rmse = float(info["verified_rmse"])

            r2["verified_rmse"] = verified_rmse
            if csv_rmse is not None:
                r2["rmse_diff"] = float(verified_rmse - csv_rmse)
            else:
                r2["rmse_diff"] = ""

            # threshold tolerance: 1e-4 is strict, 1e-3 safe
            tol = 1e-3
            if csv_rmse is None:
                r2["verify_status"] = "OK(no_csv_rmse)"
            else:
                if abs(verified_rmse - csv_rmse) <= tol:
                    r2["verify_status"] = "OK"
                else:
                    r2["verify_status"] = f"WARN(diff>{tol})"

            n_ok += 1

            if i % 20 == 0:
                print(f"[{i}/{len(rows)}] verified...")

        except Exception as e:
            r2["verified_rmse"] = ""
            r2["rmse_diff"] = ""
            r2["verify_status"] = f"ERR({repr(e)})"
            n_err += 1

        out_rows.append(r2)

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print("=" * 90)
    print("[Verify] DONE")
    print(f"[Verify] OK={n_ok}  SKIP={n_skip}  ERR={n_err}")
    print(f"[Verify] Output: {OUT_CSV}")
    print("=" * 90)


if __name__ == "__main__":
    main()
