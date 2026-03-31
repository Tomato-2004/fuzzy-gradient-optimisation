from __future__ import annotations
import torch
from .non_gradient_base import NonGradientOptimiserBase


class DEOptimizer(NonGradientOptimiserBase):
    """
    Minimal DE/rand/1/bin
    Config-aligned:
        pop_size, iters, F, CR
    """

    def __init__(
        self,
        model,
        loss_fn,
        pop_size: int,
        iters: int,
        F: float,
        CR: float,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(model=model, loss_fn=loss_fn, device=device)
        self.pop_size = int(pop_size)
        self.iters = int(iters)
        self.F = float(F)
        self.CR = float(CR)

    def optimise(self, X: torch.Tensor, y: torch.Tensor):

        base = self._flatten_params()
        dim = base.numel()

        # initial population
        pop = [base + 0.1 * torch.randn(dim) for _ in range(self.pop_size)]

        # global best
        self._load_params(base)
        best_loss = self.evaluate(X, y)
        best_state = self.snapshot()

        # print frequency: ~5 logs total
        print_every = max(1, self.iters // 5)

        for it in range(1, self.iters + 1):

            for i in range(self.pop_size):

                # pick a,b,c distinct and != i
                idxs = torch.randperm(self.pop_size).tolist()
                a, b, c = [pop[j] for j in idxs if j != i][:3]

                mutant = a + self.F * (b - c)

                # binomial crossover
                cross = torch.rand(dim) < self.CR
                trial = pop[i].clone()
                trial[cross] = mutant[cross]

                # evaluate trial
                self._load_params(trial)
                trial_loss = self.evaluate(X, y)

                # evaluate current
                self._load_params(pop[i])
                curr_loss = self.evaluate(X, y)

                # selection
                if trial_loss < curr_loss:
                    pop[i] = trial

                    if trial_loss < best_loss:
                        best_loss = trial_loss
                        self._load_params(trial)
                        best_state = self.snapshot()

            # -------- progress print --------
            if it == 1 or it % print_every == 0 or it == self.iters:
                print(f"[DE ] iter={it:03d}/{self.iters} best={best_loss:.6f}")

        self.restore(best_state)
