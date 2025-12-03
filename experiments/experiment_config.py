
CONFIG = {

    "dataset": "Airfoil_self_noise",


    "initialisation": {
        # 隶属函数生成方式（MF initializer）
        # 可选项：
        #   "heuristic"    → build_initial_fis_from_data（当前实现）
        #   （未来你可新增：如 "casp", "kmeans_mf" 等，只需在 MF_INITIALISERS 加一行）
        "mf_method": "heuristic",

        # 规则生成方式（rule generator）
        # 可选项（来自 RULE_GENERATORS）：
        #   "kmeans"       → genrules_kmeans（当前实现）
        #   （未来你可新增：如 "random", "casp_rules", "density_based" 等）
        "rule_method": "kmeans",

        # 每个输入特征使用多少个隶属函数（一般 2~7）
        "n_mf": 3,

        # 生成多少条规则（常见取值：5~30）
        "n_rules": 10,
    },

    "optimiser": {
        # 可选项（来自 OPTIMISERS）：
        #   "adam"         → train_with_adam（梯度法）
        #   "pso"          → train_with_pso（非梯度：粒子群）
        #   （未来你可以新增 "ga", "de", "sa", "cmaes" 等进化优化方法）
        "method": "pso",

        # PSO 的超参数（仅当 method = "pso" 时有效）
        "params": {
            # PSO 迭代轮数（建议：20~200）
            "epochs": 30,

            # 粒子数量（建议：10~50）
            "swarm_size": 20,

            # defuzzification 时的采样点数（越大越精确但越慢，建议 25~101）
            "point_n": 25,
        },
    },

    "adam_params": {
        # Adam 训练轮数（建议：50~500）
        "num_epochs": 100,

        # batch size（建议：16~256）
        "batch_size": 64,

        # 学习率（建议：1e-4 ~ 1e-2）
        "lr": 1e-3,

        # weight decay（一般用 0 或一个很小的值，如 1e-5）
        "weight_decay": 0.0,

        # defuzzification 时的采样点数（越大越稳定，101 是 fuzzyR 默认）
        "point_n": 101,
    },
}
