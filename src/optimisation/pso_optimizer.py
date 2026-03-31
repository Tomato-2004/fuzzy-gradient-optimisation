from __future__ import annotations
import torch
from .non_gradient_base import NonGradientOptimiserBase


class PSOOptimizer(NonGradientOptimiserBase):
    """
    Minimal PSO strictly matching config:
    pop_size, iters, w, c1, c2
    """

    def __init__(
        self,
        model,
        loss_fn,
        pop_size: int,
        iters: int,
        w: float,
        c1: float,
        c2: float,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(model=model, loss_fn=loss_fn, device=device)
        self.pop_size = int(pop_size)
        self.iters = int(iters)
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)

    def optimise(self, X: torch.Tensor, y: torch.Tensor):

        x0 = self._flatten_params()
        dim = x0.numel()

        pos = [x0 + 0.1 * torch.randn(dim) for _ in range(self.pop_size)]
        vel = [torch.zeros(dim) for _ in range(self.pop_size)]

        pbest = [p.clone() for p in pos]
        pbest_val = [float("inf")] * self.pop_size

        self._load_params(x0)
        gbest = x0.clone()
        gbest_val = self.evaluate(X, y)
        best_state = self.snapshot()

        print_every = max(1, self.iters // 5)

        for it in range(1, self.iters + 1):

            # evaluate particles
            for i in range(self.pop_size):
                self._load_params(pos[i])
                val = self.evaluate(X, y)

                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest[i] = pos[i].clone()

                if val < gbest_val:
                    gbest_val = val
                    gbest = pos[i].clone()
                    best_state = self.snapshot()

            # update velocity & position
            for i in range(self.pop_size):
                r1 = torch.rand(dim)
                r2 = torch.rand(dim)
                vel[i] = (
                    self.w * vel[i]
                    + self.c1 * r1 * (pbest[i] - pos[i])
                    + self.c2 * r2 * (gbest - pos[i])
                )
                pos[i] = pos[i] + vel[i]

            # -------- progress print --------
            if it == 1 or it % print_every == 0 or it == self.iters:
                print(f"[PSO] iter={it:03d}/{self.iters} best={gbest_val:.6f}")

        self.restore(best_state)
