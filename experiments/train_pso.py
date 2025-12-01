import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from experiments.utils.datasets import load_dataset
from src.rule_generator.genfis import build_initial_fis_from_data
from src.rule_generator.genrule import genrules_kmeans
from src.trainable_fis import TrainableFIS
from src.optimisation.pso_optimizer import train_with_pso


def train_one_dataset_pso(
    dataset_name: str,
    n_mfs_per_input: int = 3,
    n_mfs_output: int = 3,
    n_rules: int = 10,
    num_epochs: int = 30,
    swarm_size: int = 20,
    point_n: int = 25,
    random_state: int = 0,
):

    print(f"\n===== [PSO] Training dataset: {dataset_name} =====")

    X_train, X_test, y_train, y_test = load_dataset(
        dataset_name,
        test_size=0.2,
        random_state=random_state,
    )

    # 1. FIS 结构（必须和 Adam 一样）
    fis = build_initial_fis_from_data(
        X_train,
        y_train,
        n_mfs_per_input=n_mfs_per_input,
        n_mfs_output=n_mfs_output,
        fis_name=f"pso_{dataset_name}",
    )

    # 2. kmeans 规则
    genrules_kmeans(
        fis,
        X_train,
        y_train,
        n_rules=n_rules,
        weight=1.0,
        and_or=1,
        random_state=random_state,
    )

    fis.summary()

    # 3. 包装为可优化的模型
    model = TrainableFIS(fis)

    # 4. 用 PSO 训练
    model = train_with_pso(
        model,
        X_train,
        y_train,
        epochs=num_epochs,
        swarm_size=swarm_size,
        point_n=point_n,
    )

    # 5. 计算最终 MSE
    model.eval()
    with torch.no_grad():
        train_pred = model(torch.tensor(X_train, dtype=torch.float32), point_n).numpy()
        test_pred = model(torch.tensor(X_test, dtype=torch.float32), point_n).numpy()

        train_mse = float(np.mean((train_pred.reshape(-1) - y_train) ** 2))
        test_mse = float(np.mean((test_pred.reshape(-1) - y_test) ** 2))

    print(f"[PSO-{dataset_name}] Final train MSE = {train_mse:.6f}, test MSE = {test_mse:.6f}")

    return {
        "dataset": dataset_name,
        "train_mse": train_mse,
        "test_mse": test_mse,
    }
