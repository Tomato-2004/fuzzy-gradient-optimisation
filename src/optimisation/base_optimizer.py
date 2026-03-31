# src/optimisation/base_optimizer.py
from __future__ import annotations
import abc
import torch


class OptimiserBase(abc.ABC):

    def __init__(self, model, loss_fn, device="cpu"):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

    @abc.abstractmethod
    def step(self, X, y):
        pass

    def evaluate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            yhat = self.model(X)
            return float(self.loss_fn(y, yhat).item())

    def snapshot(self):
        return {
            "model": {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        }

    def restore(self, snap):
        self.model.load_state_dict(snap["model"])
