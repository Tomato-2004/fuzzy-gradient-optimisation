# src/optimisation/utils.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch


def flatten_params(model: torch.nn.Module) -> torch.Tensor:
    """Return a 1D tensor (on CPU) containing all parameters concatenated."""
    with torch.no_grad():
        flats: List[torch.Tensor] = []
        for p in model.parameters():
            flats.append(p.detach().reshape(-1).cpu())
        if len(flats) == 0:
            return torch.zeros(0, dtype=torch.float32)
        return torch.cat(flats, dim=0).to(torch.float32)


def assign_flat_params(model: torch.nn.Module, flat: torch.Tensor) -> None:
    """Assign a 1D tensor (CPU ok) back into model parameters."""
    with torch.no_grad():
        flat = flat.detach().cpu().to(torch.float32).flatten()
        idx = 0
        for p in model.parameters():
            n = p.numel()
            chunk = flat[idx: idx + n].view_as(p).to(p.device, dtype=p.dtype)
            p.copy_(chunk)
            idx += n
        if idx != flat.numel():
            raise ValueError(f"Flat size mismatch: used {idx}, got {flat.numel()}")


def as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def from_numpy(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def make_init_box(x0: torch.Tensor, init_scale: float = 0.2, clip: float = 5.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a simple search box around x0:
      lo = x0 - init_scale * (|x0| + 1)
      hi = x0 + init_scale * (|x0| + 1)
    plus optional global clipping.
    """
    x0 = x0.detach().cpu().to(torch.float32).flatten()
    span = init_scale * (x0.abs() + 1.0)
    lo = x0 - span
    hi = x0 + span
    if clip is not None:
        lo = torch.clamp(lo, -clip, clip)
        hi = torch.clamp(hi, -clip, clip)
    return lo, hi


def clip_vec(x: torch.Tensor, clip: float | None) -> torch.Tensor:
    if clip is None:
        return x
    return torch.clamp(x, -clip, clip)
