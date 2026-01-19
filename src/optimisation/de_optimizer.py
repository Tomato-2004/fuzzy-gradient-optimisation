# src/optimisation/de_optimizer.py

import numpy as np
import torch


# ========== Fitness ==========
def evaluate_fis(trainable_fis, X, y, point_n=25):
    trainable_fis.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    preds = trainable_fis(X_t, point_n=point_n).detach().numpy()
    return np.mean((preds.reshape(-1) - y)**2)


# ========== DE ==========
def train_with_de(
    trainable_fis,
    X_train,
    y_train,
    pop_size=30,
    F=0.5,
    CR=0.9,
    n_generations=50,
    point_n=25,
):


    dim = len(trainable_fis.flatten_params())

    pop = np.random.randn(pop_size, dim) * 0.1
    fitness = np.zeros(pop_size)

    def evaluate(ind):
        trainable_fis.assign_params(ind)
        return evaluate_fis(trainable_fis, X_train, y_train, point_n)

    for i in range(pop_size):
        fitness[i] = evaluate(pop[i])

    for gen in range(n_generations):

        new_pop = np.zeros_like(pop)
        new_fit = np.zeros_like(fitness)

        for i in range(pop_size):

            idxs = np.random.choice(pop_size, 3, replace=False)
            r1, r2, r3 = pop[idxs]

            v = r1 + F * (r2 - r3)

            cross_mask = np.random.rand(dim) < CR
            u = np.where(cross_mask, v, pop[i])

            fit_u = evaluate(u)
            if fit_u < fitness[i]:
                new_pop[i] = u
                new_fit[i] = fit_u
            else:
                new_pop[i] = pop[i]
                new_fit[i] = fitness[i]

        pop = new_pop
        fitness = new_fit

        print(f"[DE Gen {gen+1}/{n_generations}] best_mse = {fitness.min():.6f}")

    best = pop[fitness.argmin()]
    trainable_fis.assign_params(best)
    return trainable_fis
