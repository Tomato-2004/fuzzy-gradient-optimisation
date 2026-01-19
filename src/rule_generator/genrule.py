# src/rule_generator/genrule.py
import numpy as np
import torch
from sklearn.cluster import KMeans
from ..FuzzyInferenceSystem import FuzzyInferenceSystem


def _best_mf_index_for_value(
    fis: FuzzyInferenceSystem,
    var_type: str,
    var_index: int,
    value: float,
) -> int:
    
    if var_type == "input":
        if var_index >= len(fis.input):
            raise IndexError(f"input index {var_index} out of range")
        var = fis.input[var_index]
    elif var_type == "output":
        if var_index >= len(fis.output):
            raise IndexError(f"output index {var_index} out of range")
        var = fis.output[var_index]
    else:
        raise ValueError("var_type must be 'input' or 'output'")

    if len(var["mf"]) == 0:
        raise RuntimeError(f"No membership functions defined for {var_type}[{var_index}]")

    best_idx = 1
    best_mu = -1.0

    x = torch.tensor([value], dtype=torch.float32)

    for k, mf in enumerate(var["mf"]):
        mu = fis.evalmf(x, mf["type"], mf["params"])
        
        mu_val = float(mu.detach().cpu().numpy())
        if mu_val > best_mu:
            best_mu = mu_val
            best_idx = k + 1  

    return best_idx


def genrules_kmeans(
    fis: FuzzyInferenceSystem,
    X,
    y,
    n_rules: int,
    weight: float = 1.0,
    and_or: int = 1,
    random_state: int = 0,
):
    
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.shape[0] != X.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    data = np.concatenate([X, y], axis=1)

    if n_rules <= 0:
        raise ValueError("n_rules must be positive")

    if n_rules > data.shape[0]:
        n_rules = data.shape[0]

    km = KMeans(n_clusters=n_rules, random_state=random_state, n_init=10)
    km.fit(data)
    centers = km.cluster_centers_

    n_inputs = X.shape[1]

    for c in centers:
        x_center = c[:n_inputs]
        y_center = c[n_inputs]

        
        ante_indices = []
        for i in range(n_inputs):
            idx = _best_mf_index_for_value(fis, "input", i, x_center[i])
            ante_indices.append(idx)

        
        cons_idx = _best_mf_index_for_value(fis, "output", 0, y_center)

        rule = ante_indices + [cons_idx, weight, and_or]
        fis.add_rule(rule)
