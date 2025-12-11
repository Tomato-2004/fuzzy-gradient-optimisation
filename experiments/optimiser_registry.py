# experiments/optimiser_registry.py

from typing import Dict, Any

from src.optimisation.adam_optimizer import train_with_adam
from src.optimisation.pso_optimizer import train_with_pso
from src.optimisation.ga_optimizer import train_with_ga
from src.optimisation.de_optimizer import train_with_de   # ← 新增
from src.optimisation.cmaes_optimizer import train_with_cmaes



# ============================
# Adam
# ============================
def run_adam(model, X_train, y_train, config: Dict[str, Any]):

    p = config["adam_params"]

    trained_model, history = train_with_adam(
        model,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        num_epochs=p.get("num_epochs", 100),
        batch_size=p.get("batch_size", 64),
        lr=p.get("lr", 1e-3),
        weight_decay=p.get("weight_decay", 0.0),
        point_n=p.get("point_n", 101),
        device="cpu",
    )
    return trained_model, history


# ============================
# PSO
# ============================
def run_pso(model, X_train, y_train, config: Dict[str, Any]):

    p = config["pso_params"]

    trained_model = train_with_pso(
        model,
        X_train,
        y_train,
        epochs=p.get("epochs", 30),
        swarm_size=p.get("swarm_size", 20),
        point_n=p.get("point_n", 25),
    )
    return trained_model, None


# ============================
# GA
# ============================
def run_ga(model, X_train, y_train, config: Dict[str, Any]):

    p = config["ga_params"]

    trained_model = train_with_ga(
        model,
        X_train,
        y_train,
        pop_size=p.get("pop_size", 30),
        n_generations=p.get("n_generations", 50),
        mutation_rate=p.get("mutation_rate", 0.1),
        crossover_rate=p.get("crossover_rate", 0.7),
        point_n=p.get("point_n", 25),
    )
    return trained_model, None


# ============================
# DE
# ============================
def run_de(model, X_train, y_train, config: Dict[str, Any]):

    p = config["de_params"]

    trained_model = train_with_de(
        model,
        X_train,
        y_train,
        pop_size=p.get("pop_size", 30),
        F=p.get("F", 0.5),
        CR=p.get("CR", 0.9),
        n_generations=p.get("n_generations", 50),
        point_n=p.get("point_n", 25),
    )
    return trained_model, None

def run_cmaes(model, X_train, y_train, config):
    p = config["cmaes_params"]
    trained = train_with_cmaes(
        model,
        X_train,
        y_train,
        n_generations=p.get("n_generations", 50),
        population=p.get("population", 20),
        sigma_init=p.get("sigma_init", 0.1),
        point_n=p.get("point_n", 25),
    )
    return trained, None



# ============================
# 注册表（唯一入口）
# ============================
OPTIMISERS = {
    "adam": run_adam,
    "pso": run_pso,
    "ga": run_ga,
    "de": run_de,
    "cmaes": run_cmaes,   # ← 新增
}
