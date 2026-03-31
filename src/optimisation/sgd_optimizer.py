# src/optimisation/sgd_optimizer.py
from __future__ import annotations
import torch
from .base_optimizer import OptimiserBase


class SGDOptimizer(OptimiserBase):
    """
    SGD optimiser (Chen–Chao style GD):
    - objective: training RMSE
    - no validation / test during training
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        device: str = "cpu",
    ):
        super().__init__(model=model, loss_fn=loss_fn, device=device)

        self.opt = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    def step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.model.train()
        self.opt.zero_grad()

        yhat = self.model(X)
        loss = self.loss_fn(y, yhat)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.opt.step()

        return loss
