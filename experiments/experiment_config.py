# experiments/experiment_config.py

CONFIG = {

    # ========== dataset ==========
    "dataset": "Airfoil",


    # ========== initial methods ==========
    # mf_method:
    #   "heuristic"   
    #   "casp_single" 
    #   "casp_adapt"  
    #   "random_gauss" 
    #   "kmeans_mf" 

    #
    # rule_method:
    #   "kmeans"      ）
    #
    "initialisation": {
        "mf_method": "heuristic",
        "rule_method": "kmeans",
        "n_mf": 3,
        "n_rules": 10,
    },


    # ========== optimisers ==========
    # optimiser["method"]:
    #   "adam"   
    #   "pso"    
    #   "ga"    
    #   "de"     
    #   "cmaes"  
    #   "sgd"
    #   "rmsprop"
    "optimiser": {
        "method": "rmsprop",
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
        "point_n": 101,
    },


    # ================================
    # GA 参数
    # ================================
    "ga_params": {
        "pop_size": 30,
        "n_generations": 50,
        "mutation_rate": 0.1,
        "crossover_rate": 0.7,
        "point_n": 101,
    },


    # ================================
    # DE 参数（差分进化）
    # ================================
    "de_params": {
        "pop_size": 30,
        "F": 0.5,
        "CR": 0.9,
        "n_generations": 50,
        "point_n": 101,
    },


    # ================================
    # CMA-ES 参数
    # ================================
    "cmaes_params": {
        "n_generations": 50,
        "population": 20,
        "sigma_init": 0.1,
        "point_n": 101,
    },

    # ================================
    # RMSProp 参数（用于 gradient-only 对比实验）
    # ================================
    "rmsprop_params": {
        "num_epochs": 100,
        "batch_size": 64,
        "lr": 1e-3,
        "alpha": 0.99,
        "weight_decay": 0.0,
        "point_n": 101,
    },

    # ================================
    # SGD 参数（gradient baseline）
    # ================================
    "sgd_params": {
        "num_epochs": 100,
        "batch_size": 64,
        "lr": 1e-3,        # 👈 推荐用 1e-3
        "momentum": 0.9,
        "weight_decay": 0.0,
        "point_n": 101,
    },


}
