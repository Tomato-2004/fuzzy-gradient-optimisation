# experiments/experiment_config.py

CONFIG = {

    # ========== 数据集 ==========
    "dataset": "Airfoil",


    # ========== 初始化方法 ==========
    # 可选的 mf_method:
    #   "heuristic"   —— 原始等距 Gaussian 初始化
    #   "casp_single" —— CASP 单参数版本（sigma固定，center可训练）
    #   "casp_free"   —— CASP 完全自由（raw_sigma & raw_center 可训练）
    #   "casp_adapt"  —— CASP 自适应（center 来自分位数，sigma 来自局部密度）
    #   "random_gauss" —— 每个输入维度随机采样 center 和 sigma 的 Gaussian 初始化
    #   "kmeans_mf" —— 使用 K-Means 聚类中心生成 Gaussian MFs

    #
    # 可选的 rule_method:
    #   "kmeans"      —— K-Means 聚类规则生成（CASP 论文常用方法）
    #
    "initialisation": {
        "mf_method": "kmeans_mf",
        "rule_method": "kmeans",
        "n_mf": 3,
        "n_rules": 10,
    },


    # ========== 选择优化器 ==========
    # 可选 optimiser["method"]:
    #   "adam"   —— 梯度下降（适合可微结构）
    #   "pso"    —— 粒子群优化
    #   "ga"     —— 遗传算法
    #   "de"     —— 差分进化
    #   "cmaes"  —— 协方差矩阵自适应演化策略（适合 CASP）
    #
    "optimiser": {
        "method": "adam",
    },


    # ================================
    # Adam 参数
    # ================================
    "adam_params": {
        "num_epochs": 100,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "point_n": 101,
    },


    # ================================
    # PSO 参数
    # ================================
    "pso_params": {
        "epochs": 30,
        "swarm_size": 20,
        "point_n": 25,
    },


    # ================================
    # GA 参数
    # ================================
    "ga_params": {
        "pop_size": 30,
        "n_generations": 50,
        "mutation_rate": 0.1,
        "crossover_rate": 0.7,
        "point_n": 25,
    },


    # ================================
    # DE 参数（差分进化）
    # ================================
    "de_params": {
        "pop_size": 30,
        "F": 0.5,
        "CR": 0.9,
        "n_generations": 50,
        "point_n": 25,
    },


    # ================================
    # CMA-ES 参数
    # ================================
    "cmaes_params": {
        "n_generations": 50,
        "population": 20,
        "sigma_init": 0.1,
        "point_n": 25,
    },
}
