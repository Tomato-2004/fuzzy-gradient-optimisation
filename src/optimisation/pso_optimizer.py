# src/optimisation/pso_optimizer.py

import numpy as np
import torch


def evaluate_fis(trainable_fis, X, y, point_n=25):
    """计算 MSE（PSO 的 fitness）"""
    trainable_fis.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    preds = trainable_fis(X_t, point_n=point_n).detach().numpy()
    return np.mean((preds.reshape(-1) - y) ** 2)


def train_with_pso(
    trainable_fis,
    X_train,
    y_train,
    epochs=30,
    swarm_size=20,
    point_n=25
):
    """PSO 粒子群优化器（非梯度）"""

    # 使用 TrainableFIS 自带的参数接口
    dim = len(trainable_fis.flatten_params())

    w = 0.7
    c1 = c2 = 1.5

    # 初始化粒子
    swarm_pos = np.random.randn(swarm_size, dim) * 0.1
    swarm_vel = np.zeros((swarm_size, dim))
    pbest_pos = swarm_pos.copy()
    pbest_val = np.full(swarm_size, np.inf)

    gbest_pos = None
    gbest_val = np.inf

    for epoch in range(epochs):

        for i in range(swarm_size):
            # 写回参数
            trainable_fis.assign_params(swarm_pos[i])

            # 计算 Fitness
            val = evaluate_fis(trainable_fis, X_train, y_train, point_n)

            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest_pos[i] = swarm_pos[i].copy()

            if val < gbest_val:
                gbest_val = val
            gbest_pos = pbest_pos[pbest_val.argmin()].copy()

        # PSO 更新
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)

        swarm_vel = (
            w * swarm_vel
            + c1 * r1 * (pbest_pos - swarm_pos)
            + c2 * r2 * (gbest_pos - swarm_pos)
        )
        swarm_pos = swarm_pos + swarm_vel

        print(f"[PSO Epoch {epoch+1}/{epochs}] best_mse = {gbest_val:.6f}")

    # 写回最终参数
    trainable_fis.assign_params(gbest_pos)

    return trainable_fis
