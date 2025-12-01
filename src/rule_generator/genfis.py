# src/rule_generator/genfis.py
import numpy as np
from ..FuzzyInferenceSystem import FuzzyInferenceSystem


def build_initial_fis_from_data(
    X,
    y,
    n_mfs_per_input: int = 3,
    n_mfs_output: int = 3,
    fis_name: str = "mamdani_fis",
):
    """
    根据数据的范围生成一个初始 Mamdani FIS（Type-1）。

    设计目标：
        - 结构上对应 fuzzyR 里用 newfis + addvar + addmf 搭出来的 Mamdani 系统
        - 所有输入 / 输出 用等距划分的 Gaussian MF
        - 数据按真实 min/max 估计变量范围（不强制归一化）
        - 单输出回归问题（多输出后面再扩展）

    参数
    ----
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    n_mfs_per_input : 每个输入变量的隶属函数数量（同质）
    n_mfs_output    : 输出变量的隶属函数数量
    fis_name        : FIS 名称，方便 summary/plot

    返回
    ----
    fis : FuzzyInferenceSystem 实例（Mamdani, Type-1）
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    n_inputs = X.shape[1]

    x_mins = X.min(axis=0)
    x_maxs = X.max(axis=0)
    y_min = float(y.min())
    y_max = float(y.max())

    fis = FuzzyInferenceSystem(name=fis_name)

    # ----- 输入变量 & MFs -----
    for i in range(n_inputs):
        low, high = float(x_mins[i]), float(x_maxs[i])

        # 避免所有样本数值一样导致 sigma=0
        if np.isclose(high, low):
            high = low + 1e-3

        fis.add_variable("input", f"x{i+1}", (low, high))

        K = n_mfs_per_input
        centers = np.linspace(low, high, K)
        # 这里用一个简单的“相邻中心间距的一半”作为 sigma
        sigma = (high - low + 1e-6) / (2 * (K - 1 + 1e-6))

        for k, c in enumerate(centers):
            fis.add_mf(
                "input",
                i,
                mf_name=f"A{i+1}_{k+1}",
                mf_type="gaussmf",
                mf_params=[sigma, float(c)],
            )

    # ----- 输出变量 & MFs -----
    if np.isclose(y_max, y_min):
        y_max = y_min + 1e-3

    fis.add_variable("output", "y", (y_min, y_max))

    K_out = n_mfs_output
    centers = np.linspace(y_min, y_max, K_out)
    sigma = (y_max - y_min + 1e-6) / (2 * (K_out - 1 + 1e-6))

    for k, c in enumerate(centers):
        fis.add_mf(
            "output",
            0,
            mf_name=f"B{k+1}",
            mf_type="gaussmf",
            mf_params=[sigma, float(c)],
        )

    return fis
