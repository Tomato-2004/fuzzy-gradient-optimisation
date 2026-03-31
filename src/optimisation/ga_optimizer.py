from __future__ import annotations
import torch
from .non_gradient_base import NonGradientOptimiserBase


class GAOptimizer(NonGradientOptimiserBase):
    """
    Minimal real-valued GA strictly matching config:
    pop_size, iters, mutation_rate
    """

    def __init__(
        self,
        model,
        loss_fn,
        pop_size: int,
        iters: int,
        mutation_rate: float,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(model=model, loss_fn=loss_fn, device=device)
        self.pop_size = int(pop_size)
        self.iters = int(iters)
        self.mutation_rate = float(mutation_rate)

    def optimise(self, X: torch.Tensor, y: torch.Tensor):

        base = self._flatten_params()
        dim = base.numel()

        pop = [base + 0.1 * torch.randn(dim) for _ in range(self.pop_size)]

        self._load_params(base)
        best_loss = self.evaluate(X, y)
        best_state = self.snapshot()

        print_every = max(1, self.iters // 5)

        for it in range(1, self.iters + 1):

            scored = []
            for v in pop:
                self._load_params(v)
                scored.append((v, self.evaluate(X, y)))

            scored.sort(key=lambda t: t[1])

            if scored[0][1] < best_loss:
                best_loss = scored[0][1]
                self._load_params(scored[0][0])
                best_state = self.snapshot()

            survivors = [v for v, _ in scored[: max(2, self.pop_size // 2)]]

            new_pop = survivors.copy()
            while len(new_pop) < self.pop_size:
                p1 = survivors[torch.randint(len(survivors), (1,)).item()]
                p2 = survivors[torch.randint(len(survivors), (1,)).item()]

                mask = torch.rand(dim) < 0.5
                child = p1.clone()
                child[mask] = p2[mask]

                if torch.rand(1).item() < self.mutation_rate:
                    child = child + 0.1 * torch.randn(dim)

                new_pop.append(child)

            pop = new_pop

            # -------- progress print --------
            if it == 1 or it % print_every == 0 or it == self.iters:
                print(f"[GA ] iter={it:03d}/{self.iters} best={best_loss:.6f}")

        self.restore(best_state)
