import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.FuzzyInferenceSystem import FuzzyInferenceSystem
import torch


# ============================================================
# 1. 原版 Mamdani 测试（不修改原功能）
# ============================================================
def test_basic_fis():
    print("=== Running FIS Basic Test (Mamdani) ===")

    fis = FuzzyInferenceSystem("tipper", fis_type="mamdani", defuzz_method="centroid")

    fis.add_variable("input", "service", (0, 10))
    fis.add_mf("input", 0, "poor", "gaussmf", [1.5, 0])
    fis.add_mf("input", 0, "good", "gaussmf", [1.5, 10])

    fis.add_variable("output", "tip", (0, 30))
    fis.add_mf("output", 0, "low", "gaussmf", [4, 0])
    fis.add_mf("output", 0, "high", "gaussmf", [4, 30])

    fis.add_rule([1, 1, 1, 1])    # poor → low
    fis.add_rule([2, 2, 1, 1])    # good → high

    print("\n=== Defuzz + evalmf check ===")
    print(f"{'Method':<12s}{'Output':>12s}{'poor(7)':>12s}{'good(7)':>12s}")

    for m in ["centroid", "bisector", "mom", "som", "lom"]:
        fis.defuzz_method = m
        out = fis.eval([7.0])
        poor_mu = fis.evalmf(7, "gaussmf", [1.5, 0]).item()
        good_mu = fis.evalmf(7, "gaussmf", [1.5, 10]).item()
        print(f"{m:<12s}{out:>12.5f}{poor_mu:>12.4f}{good_mu:>12.4f}")

    print("\n=== Plot test ===")
    fis.plotmf("input", 0)
    fis.plotmf("output", 0)
    fis.plotvar(0)
    fis.plot_graph()


# ============================================================
# 2. 新增：TS / CASP 测试（完全对齐你 notebook + R 版本）
# ============================================================
def test_ts_model():
    print("\n==============================")
    print("     Running TS Model Test    ")
    print("==============================")

    fis_ts = FuzzyInferenceSystem("ts_model", fis_type="ts")

    # 输入 1
    fis_ts.add_variable("input", "x1", (0, 1))
    fis_ts.add_mf("input", 0, "low",  "gaussmf", [0.2, 0.2])
    fis_ts.add_mf("input", 0, "high", "gaussmf", [0.2, 0.8])

    # 输入 2
    fis_ts.add_variable("input", "x2", (0, 1))
    fis_ts.add_mf("input", 1, "low",  "gaussmf", [0.2, 0.2])
    fis_ts.add_mf("input", 1, "high", "gaussmf", [0.2, 0.8])

    # TS 规则（使用 dictionary）
    fis_ts.rule = [
        {   # Rule 1: low × low
            "antecedent": [0, 0],              # MF indexes
            "coeff": [0.1, 1.0, 2.0]           # y = 0.1 + 1*x1 + 2*x2
        },
        {   # Rule 2: high × high
            "antecedent": [1, 1],
            "coeff": [-0.2, -1.0, 3.0]
        }
    ]

    # ---- Forward test ----
    print("\n=== TS Forward Test ===")
    ts_out = fis_ts.eval([0.5, 0.4])    # 自动转向 eval_ts()
    print("TS Output:", float(ts_out))

    # ---- Backward (autograd) ----
    print("\n=== TS Gradient Test ===")
    x = torch.tensor([0.5, 0.4], requires_grad=True)
    y = fis_ts.eval_ts(x)
    y.backward()
    print("Gradient wrt x:", x.grad)


if __name__ == "__main__":
    test_basic_fis()   # 原版 Mamdani
    test_ts_model()    # 新增 TS
