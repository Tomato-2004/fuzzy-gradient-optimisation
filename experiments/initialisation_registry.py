# experiments/initialisation_registry.py

from typing import Dict, Any

from src.rule_generator.genfis import (
    build_fis_heuristic,
    build_fis_casp_single,
    build_fis_casp_free,
    build_fis_casp_adapt,
    build_fis_random_gauss,
    build_fis_kmeans_mf,
)

from src.rule_generator.genrule import genrules_kmeans


# ======================================================
# 所有 MF 初始化方法
# ======================================================

MF_INITIALISERS = {
    "heuristic": build_fis_heuristic,
    "casp_single": build_fis_casp_single,
    "casp_free": build_fis_casp_free,
    "casp_adapt": build_fis_casp_adapt,
    "random_gauss": build_fis_random_gauss,
    "kmeans_mf": build_fis_kmeans_mf,
}

# ======================================================
# Rule Generators
# ======================================================

def _rule_gen_kmeans(fis, X, y, n_rules: int, **kwargs):
    genrules_kmeans(fis, X, y, n_rules=n_rules)

RULE_GENERATORS = {
    "kmeans": _rule_gen_kmeans,
}


# ======================================================
# 主函数：根据配置 build FIS
# ======================================================

def build_fis_from_config(X, y, init_cfg: Dict[str, Any]):
    mf_method = init_cfg.get("mf_method", "heuristic")
    rule_method = init_cfg.get("rule_method", "kmeans")
    n_mf = init_cfg.get("n_mf", 3)
    n_rules = init_cfg.get("n_rules", 10)

    if mf_method not in MF_INITIALISERS:
        raise ValueError(f"Unknown mf_method: {mf_method}")

    if rule_method not in RULE_GENERATORS:
        raise ValueError(f"Unknown rule_method: {rule_method}")

    # ---- 1. MF 初始化 ----
    fis = MF_INITIALISERS[mf_method](X, y, n_mf=n_mf)

    # ---- 2. 规则初始化 ----
    RULE_GENERATORS[rule_method](fis, X, y, n_rules=n_rules)

    return fis
