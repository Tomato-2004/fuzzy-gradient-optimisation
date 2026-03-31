from __future__ import annotations
import torch, numpy as np
from typing import List, Callable
from src.fis_casps import *

class TrainableFIS(torch.nn.Module):
    def __init__(
        self,
        fis,
        theta_init: List[torch.Tensor],
        rules,
        num_mfs,
        decode_trapmf_fn: Callable,
        casp_mode="single",
        device="cpu",
        theta_inputs_normalised=False,
    ):
        super().__init__()
        self.fis = fis
        self.rules = rules
        self.num_mfs = num_mfs
        self.decode_trapmf_fn = decode_trapmf_fn
        self.theta_inputs_normalised = theta_inputs_normalised

        if casp_mode == "single":
            self.psi_from_theta, self.theta_from_psi = psi_from_theta_single, theta_from_psi_single
        elif casp_mode == "adapted":
            self.psi_from_theta, self.theta_from_psi = psi_from_theta_adapted, theta_from_psi_adapted
        else:
            self.psi_from_theta, self.theta_from_psi = psi_from_theta_free, theta_from_psi_free

        self.num_blocks = len(theta_init)

        psi_blocks = []
        for i, th in enumerate(theta_init):
            is_out = i == self.num_blocks - 1
            psi_blocks.append(
                self.psi_from_theta(th, normalised=(self.theta_inputs_normalised and not is_out))
            )

        self.psi = torch.nn.Parameter(torch.cat(psi_blocks), True)
        self.idx = np.cumsum([t.numel() for t in theta_init])

    def decode_theta_list(self):
        out, s = [], 0
        for i, e in enumerate(self.idx):
            is_out = i == self.num_blocks - 1
            out.append(self.theta_from_psi(
                self.psi[s:e],
                normalised=(self.theta_inputs_normalised and not is_out)
            ))
            s = e
        return out

    def forward(self, X):
        return self.fis.eval_fis(
            self.decode_theta_list(),
            self.rules,
            X,
            self.num_mfs,
            decode_trapmf_fn=self.decode_trapmf_fn,
        )
