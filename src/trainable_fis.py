# src/trainable_fis.py
import numpy as np
import torch
from torch import nn
from .FuzzyInferenceSystem import FuzzyInferenceSystem


class TrainableFIS(nn.Module):
    """
    wrape FuzzyInferenceSystem to nn.Module

    """

    def __init__(self, fis: FuzzyInferenceSystem):
        super().__init__()
        self.fis = fis

        params = []

        
        for var_list in (self.fis.input, self.fis.output):
            for var in var_list:
                for mf in var["mf"]:
                    p = nn.Parameter(torch.as_tensor(mf["params"], dtype=torch.float32))
                    mf["params"] = p
                    params.append(p)

        
        self.mf_params = nn.ParameterList(params)

    # ============================================================
    # flatten_params() / assign_params()
    # ============================================================

    def flatten_params(self):
       
        vec = []
        for p in self.mf_params:
            vec.append(p.data.detach().cpu().numpy().reshape(-1))
        return np.concatenate(vec)

    def assign_params(self, vec):
        
        idx = 0
        for p in self.mf_params:
            n = p.numel()
            block = vec[idx: idx+n]
            idx += n
            p.data = torch.tensor(block, dtype=torch.float32).view_as(p.data)

    # ============================================================

    def forward(self, x: torch.Tensor, point_n: int = 101) -> torch.Tensor:
        """
        x: (batch_size, n_inputs) 或 (n_inputs,)
        返回: (batch_size,) tensor
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        return self.fis.eval_batch(x, point_n=point_n)
