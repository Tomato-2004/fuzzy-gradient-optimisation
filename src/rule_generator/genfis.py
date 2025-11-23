# src/rule_generator/genfis.py
import numpy as np
from ..FuzzyInferenceSystem import FuzzyInferenceSystem


def build_initial_fis_from_data(
    X,
    y,
    n_mfs_per_input=3,
    n_mfs_output=3,
    fis_name="mamdani_fis",
):
    """
    根据数据的范围生成一个初始 Mamdani FIS。
    - 所有输入 / 输出 先用等距划分的 Gaussian MF
    - 数据已经被 MinMaxScaler 归一化到 [0,1]，但这里还是按真实 min/max 做一下范围估计
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    n_inputs = X.shape[1]

    x_mins = X.min(axis=0)
    x_maxs = X.max(axis=0)
    y_min = float(y.min())
    y_max = float(y.max())

    fis = FuzzyInferenceSystem(name=fis_name)

    # ----- 输入变量 & MF -----
    for i in range(n_inputs):
        low, high = float(x_mins[i]), float(x_maxs[i])
        fis.add_variable("input", f"x{i+1}", (low, high))

        K = n_mfs_per_input
        centers = np.linspace(low, high, K)
        sigma = (high - low + 1e-6) / (2 * (K - 1 + 1e-6))

        for k, c in enumerate(centers):
            fis.add_mf("input", i,
                       mf_name=f"A{i+1}_{k+1}",
                       mf_type="gaussmf",
                       mf_params=[sigma, c])

    # ----- 输出变量 & MF -----
    fis.add_variable("output", "y", (y_min, y_max))

    K_out = n_mfs_output
    centers = np.linspace(y_min, y_max, K_out)
    sigma = (y_max - y_min + 1e-6) / (2 * (K_out - 1 + 1e-6))

    for k, c in enumerate(centers):
        fis.add_mf("output", 0,
                   mf_name=f"B{k+1}",
                   mf_type="gaussmf",
                   mf_params=[sigma, c])

    return fis
