
from typing import Dict, Any

from src.rule_generator.genfis import build_initial_fis_from_data
from src.rule_generator.genrule import genrules_kmeans



def _mf_init_heuristic(X, y, n_mf: int):
    """
    使用 build_initial_fis_from_data:
        - 输入每个变量 n_mf 个 Gaussian MF
        - 输出变量 n_mf 个 Gaussian MF
    跟你原先 basic 实验中的做法一致。
    """
    fis = build_initial_fis_from_data(
        X,
        y,
        n_mfs_per_input=n_mf,
        n_mfs_output=n_mf,
    )
    return fis


MF_INITIALISERS = {
    "heuristic": _mf_init_heuristic,

}



def _rule_gen_kmeans(fis, X, y, n_rules: int, **kwargs):
  
    genrules_kmeans(
        fis,
        X,
        y,
        n_rules=n_rules,
        
    )


RULE_GENERATORS = {
    "kmeans": _rule_gen_kmeans,

}



def build_fis_from_config(X, y, init_cfg: Dict[str, Any]):
 
    mf_method = init_cfg.get("mf_method", "heuristic")
    rule_method = init_cfg.get("rule_method", "kmeans")
    n_mf = init_cfg.get("n_mf", 3)
    n_rules = init_cfg.get("n_rules", 10)

    if mf_method not in MF_INITIALISERS:
        raise ValueError(f"Unknown mf_method: {mf_method}")

    if rule_method not in RULE_GENERATORS:
        raise ValueError(f"Unknown rule_method: {rule_method}")

    fis = MF_INITIALISERS[mf_method](X, y, n_mf=n_mf)

    RULE_GENERATORS[rule_method](fis, X, y, n_rules=n_rules)

    return fis
