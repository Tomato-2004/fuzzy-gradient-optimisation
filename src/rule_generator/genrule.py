# src/rule_generator/genrule.py
import numpy as np
import torch
from sklearn.cluster import KMeans
from ..FuzzyInferenceSystem import FuzzyInferenceSystem


def _best_mf_index_for_value(fis: FuzzyInferenceSystem, var_type: str, var_index: int, value: float) -> int:
    """
    给定一个变量的数值，找到隶属度最大的 MF index（1-based）。
    """
    if var_type == "input":
        var = fis.input[var_index]
    else:
        var = fis.output[var_index]

    best_idx = 1
    best_mu = -1.0

    x = torch.tensor([value], dtype=torch.float32)

    for k, mf in enumerate(var["mf"]):
        mu = fis.evalmf(x, mf["type"], mf["params"])
        mu_val = float(mu.detach().cpu().numpy())
        if mu_val > best_mu:
            best_mu = mu_val
            best_idx = k + 1  # 1-based

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
    """
    CASP 风格的规则生成（简化版）：
        1. 用 [X, y] 在特征空间上做 k-means 聚类，k = n_rules
        2. 对每个 cluster center：
            - 对每个输入 x_i：选择隶属度最大的 MF 作为前件
            - 对输出 y：选择隶属度最大的 MF 作为后件
        3. 构造一条 Mamdani 规则 [ante..., cons, weight, and_or]

    注意：这里假定单输出变量 fis.output[0]。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    data = np.concatenate([X, y], axis=1)

    km = KMeans(n_clusters=n_rules, random_state=random_state, n_init=10)
    km.fit(data)
    centers = km.cluster_centers_

    n_inputs = X.shape[1]

    for c in centers:
        x_center = c[:n_inputs]
        y_center = c[n_inputs]

        # 前件：每个输入一个 MF index
        ante_indices = []
        for i in range(n_inputs):
            idx = _best_mf_index_for_value(fis, "input", i, x_center[i])
            ante_indices.append(idx)

        # 后件：输出 MF index
        cons_idx = _best_mf_index_for_value(fis, "output", 0, y_center)

        rule = ante_indices + [cons_idx, weight, and_or]
        fis.add_rule(rule)
