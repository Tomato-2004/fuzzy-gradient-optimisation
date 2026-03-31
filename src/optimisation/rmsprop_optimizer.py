# src/optimisation/rmsprop_optimizer.py
from __future__ import annotations
import torch
from .base_optimizer import OptimiserBase


class RMSPropOptimizer(OptimiserBase):
    """
    RMSProp optimiser (Chen–Chao style GD)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        lr: float,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        device: str = "cpu",
    ):
        super().__init__(model=model, loss_fn=loss_fn, device=device)

        self.opt = torch.optim.RMSprop(
            self.model.parameters(),
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )

    def step(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.model.train()
        self.opt.zero_grad()

        yhat = self.model(X)
        loss = self.loss_fn(y, yhat)

        loss.backward()
        self.opt.step()

        return loss
