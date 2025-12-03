
from typing import Dict, Any

from src.optimisation.adam_optimizer import train_with_adam
from src.optimisation.pso_optimizer import train_with_pso


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


def run_pso(model, X_train, y_train, config: Dict[str, Any]):
   
    p = config["optimiser"]["params"]

    trained_model = train_with_pso(
        model,
        X_train,
        y_train,
        epochs=p.get("epochs", 30),
        swarm_size=p.get("swarm_size", 20),
        point_n=p.get("point_n", 25),
    )

    # PSO 没有 history，这里统一返回 None
    return trained_model, None


OPTIMISERS = {
    "adam": run_adam,
    "pso": run_pso,
}
