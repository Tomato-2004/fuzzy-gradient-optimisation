# src/optimisation/ga_optimizer.py

import numpy as np
import torch


# =======================
# Fitness 评估
# =======================
def evaluate_fis(trainable_fis, X, y, point_n=25):
    trainable_fis.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    preds = trainable_fis(X_t, point_n=point_n).detach().numpy()
    return np.mean((preds.reshape(-1) - y)**2)


# =======================
# GA 优化器
# =======================
def train_with_ga(
    trainable_fis,
    X_train,
    y_train,
    pop_size=30,
    n_generations=50,
    mutation_rate=0.1,
    crossover_rate=0.7,
    point_n=25,
):

    dim = len(trainable_fis.flatten_params())

    # 初始化种群（每个个体是一个参数向量）
    pop = np.random.randn(pop_size, dim) * 0.1

    # 个体评估
    def fitness(ind):
        trainable_fis.assign_params(ind)
        return evaluate_fis(trainable_fis, X_train, y_train, point_n)

    scores = np.array([fitness(ind) for ind in pop])

    for gen in range(n_generations):

        # 排序（从低 mse 到高 mse）
        idx = scores.argsort()
        pop = pop[idx]
        scores = scores[idx]

        # 保留精英
        elites = pop[:2]

        # 新种群
        new_pop = elites.copy()

        while len(new_pop) < pop_size:

            # 交叉
            if np.random.rand() < crossover_rate:
                p1, p2 = pop[np.random.randint(pop_size)], pop[np.random.randint(pop_size)]
                mask = np.random.rand(dim) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = pop[np.random.randint(pop_size)].copy()

            # 变异
            if np.random.rand() < mutation_rate:
                child += 0.1 * np.random.randn(dim)

            new_pop = np.vstack([new_pop, child])

        pop = new_pop
        scores = np.array([fitness(ind) for ind in pop])

        print(f"[GA Gen {gen+1}/{n_generations}] best_mse = {scores.min():.6f}")

    best = pop[scores.argmin()]
    trainable_fis.assign_params(best)
    return trainable_fis
