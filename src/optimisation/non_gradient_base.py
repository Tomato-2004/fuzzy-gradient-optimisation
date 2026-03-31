from __future__ import annotations
import copy
import torch


class NonGradientOptimiserBase:
    """
    Chen–Chao style non-gradient optimiser:
    - objective = training RMSE
    - NO test/val inside optimise
    - provides snapshot/restore/evaluate for subclasses
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn,
        device: str = "cpu",
        **kwargs,   # 允许 run_experiment 传多余字段时不炸（但我们不会用）
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

    # -----------------------------
    # shared utils
    # -----------------------------
    def snapshot(self):
        return copy.deepcopy(self.model.state_dict())

    def restore(self, state):
        self.model.load_state_dict(state)

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            yhat = self.model(X)
            return float(self.loss_fn(y, yhat).item())

    # -----------------------------
    # parameter vector helpers
    # -----------------------------
    def _flatten_params(self) -> torch.Tensor:
        return torch.cat([p.detach().flatten() for p in self.model.parameters()])

    def _load_params(self, vec: torch.Tensor):
        idx = 0
        with torch.no_grad():
            for p in self.model.parameters():
                n = p.numel()
                p.copy_(vec[idx:idx + n].view_as(p))
                idx += n

    # -----------------------------
    # required interface
    # -----------------------------
    def optimise(self, X: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError
