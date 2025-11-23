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


if __name__ == "__main__":
    test_basic_fis()   # 原版 Mamdani

