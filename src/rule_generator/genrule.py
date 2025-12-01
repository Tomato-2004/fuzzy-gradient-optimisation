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
    """
    给定一个变量的数值，找到隶属度最大的 MF index（1-based）。

    用于构造规则前件/后件：
        - var_type = "input"  时，从 fis.input[var_index].mf 里找
        - var_type = "output" 时，从 fis.output[var_index].mf 里找

    ⚠ 这里不需要保持梯度，因为规则生成是在训练前的“结构设计阶段”，
      仅用来确定哪一个 MF 被选作前件/后件。
    """
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
        # 这里 detach 转 float 是安全的：生成规则不参与反向传播
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
    CASP 论文风格的规则生成（简化版）：

        1. 在联合空间 [X, y] 上做 k-means 聚类，簇数 k = n_rules
        2. 对每个 cluster center：
             - 对每个输入 x_i：在对应输入变量的所有 MF 中选隶属度最大的，作为前件
             - 对输出 y  ：在输出变量所有 MF 中选隶属度最大的，作为后件
        3. 构造一条 Mamdani 规则：
             [ante_1, ..., ante_n_inputs, cons, weight, and_or]

    说明：
        - 这里假定是单输出：fis.output[0]
        - and_or = 1 表示 AND (min)，2 表示 OR (max)
        - 生成的规则直接通过 fis.add_rule 写回 FIS
    """
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
        # 安全起见：规则数不能超过样本数，否则 k-means 会报错
        n_rules = data.shape[0]

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

        # 后件：输出 MF index（单输出 fis.output[0]）
        cons_idx = _best_mf_index_for_value(fis, "output", 0, y_center)

        rule = ante_indices + [cons_idx, weight, and_or]
        fis.add_rule(rule)
