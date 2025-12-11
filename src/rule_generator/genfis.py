# src/rule_generator/genfis.py

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from ..FuzzyInferenceSystem import FuzzyInferenceSystem


# ============================================================
# 工具：创建空 FIS（只含输入输出变量，无 MF）
# ============================================================

def create_empty_fis(X, y, name="fis_init"):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)

    fis = FuzzyInferenceSystem(name=name)

    # 输入变量
    n_inputs = X.shape[1]
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    for i in range(n_inputs):
        fis.add_variable("input", f"x{i+1}", (float(mins[i]), float(maxs[i])))

    # 输出变量
    low, high = float(y.min()), float(y.max())
    fis.add_variable("output", "y", (low, high))

    return fis


# ============================================================
# Heuristic Initialization（你的原版）
# ============================================================

def build_fis_heuristic(X, y, n_mf=3):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)

    fis = create_empty_fis(X, y, "heuristic_fis")
    n_inputs = X.shape[1]

    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    # 输入 MF
    for i in range(n_inputs):
        low, high = float(mins[i]), float(maxs[i])
        centers = np.linspace(low, high, n_mf)
        sigma = (high - low + 1e-6) / (2 * (n_mf - 1 + 1e-6))

        for k, c in enumerate(centers):
            fis.add_mf("input", i, f"A{i+1}_{k+1}", "gaussmf", [sigma, float(c)])

    # 输出 MF
    ymin, ymax = float(y.min()), float(y.max())
    centers = np.linspace(ymin, ymax, n_mf)
    sigma = (ymax - ymin + 1e-6) / (2 * (n_mf - 1 + 1e-6))

    for k, c in enumerate(centers):
        fis.add_mf("output", 0, f"B{k+1}", "gaussmf", [sigma, float(c)])

    return fis


# ============================================================
# CASP Single
# ============================================================

def build_fis_casp_single(X, y, n_mf=3):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)

    fis = create_empty_fis(X, y, "casp_single_fis")

    n_inputs = X.shape[1]
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    for i in range(n_inputs):
        low, high = float(mins[i]), float(maxs[i])

        centers = np.linspace(low, high, n_mf)
        sigma = (high - low + 1e-6) / (2 * n_mf)

        for k in range(n_mf):
            params = torch.nn.Parameter(torch.tensor([sigma, float(centers[k])]))
            fis.add_mf("input", i, f"A{i+1}_{k+1}", "gaussmf_casp_single", params)

    # 输出
    ymin, ymax = float(y.min()), float(y.max())
    centers = np.linspace(ymin, ymax, n_mf)
    sigma = (ymax - ymin + 1e-6) / (2 * n_mf)

    for k in range(n_mf):
        params = torch.nn.Parameter(torch.tensor([sigma, float(centers[k])]))
        fis.add_mf("output", 0, f"B{k+1}", "gaussmf_casp_single", params)

    return fis


# ============================================================
# CASP Free
# ============================================================

def build_fis_casp_free(X, y, n_mf=3):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)

    fis = create_empty_fis(X, y, "casp_free_fis")

    n_inputs = X.shape[1]
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    for i in range(n_inputs):
        low, high = float(mins[i]), float(maxs[i])
        centers = np.linspace(low, high, n_mf)

        for k in range(n_mf):
            raw_sigma = torch.tensor(0.1 * (high - low), dtype=torch.float32)
            raw_center = torch.tensor(float(centers[k]), dtype=torch.float32)
            params = torch.nn.Parameter(torch.stack([raw_sigma, raw_center]))
            fis.add_mf("input", i, f"A{i+1}_{k+1}", "gaussmf_casp_free", params)

    ymin, ymax = float(y.min()), float(y.max())
    centers = np.linspace(ymin, ymax, n_mf)

    for k in range(n_mf):
        raw_sigma = torch.tensor(0.1 * (ymax - ymin), dtype=torch.float32)
        raw_center = torch.tensor(float(centers[k]), dtype=torch.float32)
        params = torch.nn.Parameter(torch.stack([raw_sigma, raw_center]))
        fis.add_mf("output", 0, f"B{k+1}", "gaussmf_casp_free", params)

    return fis


# ============================================================
# CASP Adaptive
# ============================================================

def build_fis_casp_adapt(X, y, n_mf=3):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)

    fis = create_empty_fis(X, y, "casp_adapt_fis")

    n_inputs = X.shape[1]
    for i in range(n_inputs):
        vals = X[:, i]
        centers = np.quantile(vals, np.linspace(0, 1, n_mf))
        diffs = np.diff(centers)
        sigma = diffs.mean() if len(diffs) > 0 else (vals.max() - vals.min()) / n_mf

        for k in range(n_mf):
            raw_sigma = torch.tensor(float(sigma), dtype=torch.float32)
            raw_center = torch.tensor(float(centers[k]), dtype=torch.float32)
            params = torch.nn.Parameter(torch.stack([raw_sigma, raw_center]))
            fis.add_mf("input", i, f"A{i+1}_{k+1}", "gaussmf_casp_adapt", params)

    # 输出
    centers = np.quantile(y, np.linspace(0, 1, n_mf))
    diffs = np.diff(centers)
    sigma = diffs.mean() if len(diffs) > 0 else (y.max() - y.min()) / n_mf

    for k in range(n_mf):
        raw_sigma = torch.tensor(float(sigma), dtype=torch.float32)
        raw_center = torch.tensor(float(centers[k]), dtype=torch.float32)
        params = torch.nn.Parameter(torch.stack([raw_sigma, raw_center]))
        fis.add_mf("output", 0, f"B{k+1}", "gaussmf_casp_adapt", params)

    return fis


# ============================================================
# Random Gaussian MF（新增）
# ============================================================

def build_fis_random_gauss(X, y, n_mf=3):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1)

    fis = create_empty_fis(X, y, "random_gauss_fis")

    n_inputs = X.shape[1]
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    for i in range(n_inputs):
        low, high = float(mins[i]), float(maxs[i])

        for k in range(n_mf):
            center = np.random.uniform(low, high)
            sigma = np.random.uniform((high-low)/20, (high-low)/5)
            fis.add_mf("input", i, f"A{i+1}_{k+1}", "gaussmf", [sigma, center])

    # 输出随机
    ymin, ymax = float(y.min()), float(y.max())
    for k in range(n_mf):
        center = np.random.uniform(ymin, ymax)
        sigma = np.random.uniform((ymax-ymin)/20, (ymax-ymin)/5)
        fis.add_mf("output", 0, f"B{k+1}", "gaussmf", [sigma, center])

    return fis


# ============================================================
# KMeans MF（新增）
# ============================================================

def build_fis_kmeans_mf(X, y, n_mf=3):
    fis = create_empty_fis(X, y, "kmeans_mf_fis")
    X = np.asarray(X, float)

    n_inputs = X.shape[1]

    # 每个特征独立 KMeans
    for i in range(n_inputs):
        vals = X[:, i].reshape(-1, 1)
        km = KMeans(n_clusters=n_mf, n_init=10).fit(vals)
        centers = np.sort(km.cluster_centers_.flatten())

        # sigma = 距离最近中心的平均值
        diffs = np.diff(centers)
        sigma = diffs.mean() if len(diffs) else 1.0

        for k in range(n_mf):
            fis.add_mf("input", i, f"A{i+1}_{k+1}", "gaussmf", [sigma, float(centers[k])])

    # 输出也用 KMeans
    y_vals = y.reshape(-1, 1)
    km = KMeans(n_clusters=n_mf, n_init=10).fit(y_vals)
    centers = np.sort(km.cluster_centers_.flatten())

    diffs = np.diff(centers)
    sigma = diffs.mean() if len(diffs) else 1.0

    for k in range(n_mf):
        fis.add_mf("output", 0, f"B{k+1}", "gaussmf", [sigma, float(centers[k])])

    return fis
