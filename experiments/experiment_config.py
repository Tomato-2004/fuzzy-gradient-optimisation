from __future__ import annotations

# ======================================================
# Global
# ======================================================
SEED = 0
DEVICE = "cpu"

# ======================================================
# Experiment control
# ======================================================
DATASET = "airfoil"  
FOLD = 1
INIT_SOURCE = "kmeans"
OPTIMISER_NAME = "cmaes"

# ======================================================
# CASPs
# ======================================================
CASP_MODE = "free"   # "single" | "adapted" | "free"

# ======================================================
# Training schedule
# ======================================================
EPOCHS = 300
LOG_EPOCHS = [1,10, 50, 100, 150, 200, 250, 300]

# ======================================================
# Dataset registry
# ======================================================
DATASET_CONFIG = {

    
    "airfoil": {
        "use_pca": False,
        "num_mfs": [7,5,5,3,7,3],
        "theta_inputs_normalised": True,    
        "input_range_normalised": True,     
    },

    "autompg6": {
        "use_pca": False,
        "num_mfs": [3,4,4,4,4,9],
        "theta_inputs_normalised": False,
        "input_range_normalised": False,
    },

    "laser": {
        "use_pca": False,
        "num_mfs": [5, 9, 9, 9, 9],
        "theta_inputs_normalised": False,  
        "input_range_normalised": False,    
    },

    # ---------- PCA ----------
    "abalone": {
        "use_pca": True,
        "num_mfs": [3,3,4,7,4,6],
        "theta_inputs_normalised": False,
        "input_range_normalised": False,
    },

    "concrete": {
        "use_pca": True,
        "num_mfs": [4,3,4,4,4,9],
        "theta_inputs_normalised": True,
        "input_range_normalised": True,
    },

    "ankara": {
        "use_pca": True,
        "num_mfs": [3,3,3,3,3,8],
        "theta_inputs_normalised": True,
        "input_range_normalised": True,
    },

    "izmir": {
        "use_pca": True,
        "num_mfs": [3,8,4,3,3,9],
        "theta_inputs_normalised": True,
        "input_range_normalised": True,
    },

    "baseball": {
        "use_pca": True,
        "num_mfs": [3,3,3,6,3,9],
        "theta_inputs_normalised": True,
        "input_range_normalised": True,
    },

    "treasury": {
        "use_pca": True,
        "num_mfs": [9,6,9,9,7,6],
        "theta_inputs_normalised": True,
        "input_range_normalised": True,
    },

    "wine": {
        "use_pca": True,
        "num_mfs": [4,4,3,3,3,3],
        "theta_inputs_normalised": True,
        "input_range_normalised": True,
    },
}

# ======================================================
# Optimiser hyperparameters
# ======================================================
OPTIMISER_CONFIG = {
    "adam": {"lr": 0.03, "amsgrad": False, "weight_decay": 0.0},
    "sgd": {"lr": 0.003, "momentum": 0.9, "weight_decay": 0.0, "nesterov": False},
    "rmsprop": {"lr": 0.001, "alpha": 0.99, "weight_decay": 0.0, "momentum": 0.0},
    "pso": {"pop_size": 30, "iters": 10, "w": 0.9, "c1": 1.2, "c2": 1.2},
    "ga": {"pop_size": 30, "iters": 10, "mutation_rate": 0.05},
    "de": {"pop_size": 30, "iters": 10, "F": 0.5, "CR": 0.9},
    "cmaes": {"pop_size": 30, "iters": 10, "sigma": 0.1},
}
