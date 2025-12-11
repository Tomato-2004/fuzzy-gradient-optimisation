# src/optimisation/cmaes_optimizer.py

import numpy as np
import torch


# ---------------------------------
# Fitness function
# ---------------------------------
def evaluate_fis(trainable_fis, X, y, point_n=25):
    trainable_fis.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    preds = trainable_fis(X_t, point_n=point_n).detach().numpy()
    return np.mean((preds.reshape(-1) - y) ** 2)


# ---------------------------------
# CMA-ES Optimizer
# ---------------------------------
def train_with_cmaes(
    trainable_fis,
    X_train,
    y_train,
    n_generations=50,
    population=20,
    sigma_init=0.1,
    point_n=25,
):
    """
    CMA-ES continuous optimization
    - Very powerful for high-dimensional FIS optimization
    """

    dim = len(trainable_fis.flatten_params())

    # Initial mean of search distribution
    mean = np.random.randn(dim) * 0.1  
    sigma = sigma_init

    # CMA-ES parameters
    lambda_ = population
    mu = lambda_ // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mueff = (weights.sum() ** 2) / np.sum(weights ** 2)

    # Adaptation constants
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

    # Evolution paths & covariance
    pc = np.zeros(dim)
    ps = np.zeros(dim)
    C = np.eye(dim)

    for gen in range(n_generations):

        # Sampling
        z_samples = np.random.randn(lambda_, dim)
        y_samples = z_samples @ np.linalg.cholesky(C).T
        x_samples = mean + sigma * y_samples

        # Evaluate fitness
        fitness = []
        for i in range(lambda_):
            trainable_fis.assign_params(x_samples[i])
            fitness.append(evaluate_fis(trainable_fis, X_train, y_train, point_n))
        fitness = np.array(fitness)

        # Rank best individuals
        idx = np.argsort(fitness)
        best_idx = idx[:mu]
        best_y = y_samples[best_idx]
        best_x = x_samples[best_idx]

        # Print log
        print(f"[CMA-ES Gen {gen+1}/{n_generations}] best_mse = {fitness.min():.6f}")

        # Recompute mean
        old_mean = mean.copy()
        mean = np.sum(best_x * weights[:, None], axis=0)

        # Path update
        y_w = np.sum(best_y * weights[:, None], axis=0)
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_w

        # Covariance matrix update
        hsig = int((np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen+1)))) < (1.4 + 2 / (dim + 1)))
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w

        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * c1 * C)

        for i in range(mu):
            C += cmu * weights[i] * np.outer(best_y[i], best_y[i])

        sigma *= np.exp((np.linalg.norm(ps) - np.sqrt(dim)) / (np.sqrt(dim) * damps))

    # Assign final parameters
    trainable_fis.assign_params(mean)
    return trainable_fis
