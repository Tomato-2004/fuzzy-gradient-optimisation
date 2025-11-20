# experiments/optimizers/adam_optimizer.py

import numpy as np

def adam_step(params, grads, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, state=None):
    """Minimal Adam for basic experiment."""

    if state is None:
        state = {
            "m": [np.zeros_like(p) for p in params],
            "v": [np.zeros_like(p) for p in params],
            "t": 0
        }

    state["t"] += 1

    new_params = []

    for i, (p, g) in enumerate(zip(params, grads)):

        m = state["m"][i]
        v = state["v"][i]

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)

        m_hat = m / (1 - beta1 ** state["t"])
        v_hat = v / (1 - beta2 ** state["t"])

        p = p - lr * m_hat / (np.sqrt(v_hat) + eps)

        state["m"][i] = m
        state["v"][i] = v

        new_params.append(p)

    return new_params, state
