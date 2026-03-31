from __future__ import annotations
import torch
from .non_gradient_base import NonGradientOptimiserBase


class CMAESOptimizer(NonGradientOptimiserBase):
    """
    Minimal CMA-ES-like ES (diagonal covariance)
    Config: pop_size, iters, sigma
    """

    def __init__(
        self,
        model,
        loss_fn,
        pop_size: int,
        iters: int,
        sigma: float,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(model=model, loss_fn=loss_fn, device=device)
        self.pop_size = int(pop_size)
        self.iters = int(iters)
        self.sigma = float(sigma)

    def optimise(self, X: torch.Tensor, y: torch.Tensor):

        mean = self._flatten_params()
        dim = mean.numel()
        diag = torch.ones(dim)

        self._load_params(mean)
        best_loss = self.evaluate(X, y)
        best_state = self.snapshot()

        print_every = max(1, self.iters // 5)

        for it in range(1, self.iters + 1):

            eps = torch.randn(self.pop_size, dim)
            samples = mean.unsqueeze(0) + self.sigma * eps * diag.unsqueeze(0)

            scored = []
            for s in samples:
                self._load_params(s)
                scored.append((s, self.evaluate(X, y)))
            scored.sort(key=lambda t: t[1])

            mu = max(1, self.pop_size // 2)
            mean = torch.stack([v for v, _ in scored[:mu]]).mean(0)

            spread = torch.stack([v for v, _ in scored[:mu]]).std(0)
            diag = torch.clamp(spread, min=1e-6)

            if scored[0][1] < best_loss:
                best_loss = scored[0][1]
                self._load_params(scored[0][0])
                best_state = self.snapshot()

            # -------- progress print --------
            if it == 1 or it % print_every == 0 or it == self.iters:
                print(f"[CMA] iter={it:03d}/{self.iters} best={best_loss:.6f}")

        self.restore(best_state)
