# src/optimisation/adam_optimizer.py
from __future__ import annotations
import torch
from .base_optimizer import OptimiserBase


class AdamOptimizer(OptimiserBase):

    def __init__(self, model, loss_fn, lr, amsgrad=True, weight_decay=0.0, device="cpu"):
        super().__init__(model, loss_fn, device)
        self.opt = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            amsgrad=amsgrad,
            weight_decay=weight_decay,
        )

    def step(self, X, y):
        self.model.train()
        self.opt.zero_grad()
        yhat = self.model(X)
        loss = self.loss_fn(y, yhat)
        loss.backward()
        self.opt.step()
        return loss
