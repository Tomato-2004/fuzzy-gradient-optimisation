# src/fis_casps.py
from __future__ import annotations
import torch
import torch.nn.functional as F

# ======================================================
# Constants (与 R 代码保持一致)
# ======================================================

EPS = 1e-12
INF = 1e6


# ======================================================
# CASPs-Single (casp_v2 in R)
# calculate_θ_casp_v2 / calculate_ψ_casp_v2
# ======================================================

def theta_from_psi_single(psi: torch.Tensor, normalised: bool = False) -> torch.Tensor:
    psi = psi.flatten()
    n = psi.numel()
    theta = torch.zeros_like(psi)

    # csum(exp(psi[2:n])) in R (1-indexed)
    csum = torch.cumsum(torch.exp(psi[1:]), dim=0)
    csum = torch.cat([torch.zeros(1, device=psi.device), csum], dim=0)

    theta[0] = psi[0]
    theta[1] = psi[0] + csum[2]

    if n > 4:
        J = n // 4 - 1
        for j in range(1, J + 1):
            theta[4*j - 2] = psi[0] + csum[4*j - 3]
            theta[4*j - 1] = psi[0] + csum[4*j - 1]
            theta[4*j + 0] = psi[0] + csum[4*j + 0]
            theta[4*j + 1] = psi[0] + csum[4*j + 2]

    theta[n - 2] = psi[0] + csum[n - 3]
    theta[n - 1] = psi[0] + csum[n - 1]

    if normalised:
        theta = torch.sigmoid(theta)

    return theta


def psi_from_theta_single(theta: torch.Tensor, normalised: bool = False) -> torch.Tensor:
    theta = theta.flatten()
    n = theta.numel()
    psi = torch.zeros_like(theta)

    if normalised:
        theta = torch.clamp(theta, min=EPS, max=1.0 - EPS)
        theta = torch.log(theta / (1 - theta))


    psi[0] = theta[0]
    psi[1] = torch.log(torch.clamp(theta[2] - theta[0], min=EPS))

    if n > 4:
        J = n // 4 - 1
        for j in range(1, J + 1):
            psi[4*j - 2] = torch.log(torch.clamp(theta[4*j - 3] - theta[4*j - 2], min=EPS))
            psi[4*j - 1] = torch.log(torch.clamp(theta[4*j - 1] - theta[4*j - 3], min=EPS))
            psi[4*j + 0] = torch.log(torch.clamp(theta[4*j + 0] - theta[4*j - 1], min=EPS))
            psi[4*j + 1] = torch.log(torch.clamp(theta[4*j + 2] - theta[4*j + 0], min=EPS))

    psi[n - 2] = torch.log(torch.clamp(theta[n - 3] - theta[n - 2], min=EPS))
    psi[n - 1] = torch.log(torch.clamp(theta[n - 1] - theta[n - 3], min=EPS))

    return psi


# ======================================================
# CASPs-Adapted (calculate_θ_casps_adapted)
# allow overlap among more categories
# ======================================================

def theta_from_psi_adapted(psi: torch.Tensor, normalised: bool = False) -> torch.Tensor:
    psi = psi.flatten()
    n = psi.numel()
    theta = torch.zeros_like(psi)

    theta[0] = psi[0]
    theta[1] = psi[0] + torch.exp(psi[1])

    if n > 4:
        J = n // 4 - 1
        for j in range(1, J + 1):
            theta[4*j - 2] = theta[4*j - 3] - torch.exp(psi[4*j - 2])
            theta[4*j - 1] = theta[4*j - 2] + torch.exp(psi[4*j - 1])
            theta[4*j + 0] = theta[4*j - 1] + torch.exp(psi[4*j + 0])
            theta[4*j + 1] = theta[4*j + 0] + torch.exp(psi[4*j + 1])

    theta[n - 2] = theta[n - 3] - torch.exp(psi[n - 2])
    theta[n - 1] = theta[n - 2] + torch.exp(psi[n - 1])

    if normalised:
        theta = torch.sigmoid(theta)

    return theta


def psi_from_theta_adapted(theta: torch.Tensor, normalised: bool = False) -> torch.Tensor:
    theta = theta.flatten()
    n = theta.numel()
    psi = torch.zeros_like(theta)

    if normalised:
        theta = torch.log(theta / (1 - theta))

    psi[0] = theta[0]
    psi[1] = torch.log(torch.clamp(theta[1] - theta[0], min=EPS))

    if n > 4:
        J = n // 4 - 1
        for j in range(1, J + 1):
            psi[4*j - 2] = torch.log(torch.clamp(theta[4*j - 3] - theta[4*j - 2], min=EPS))
            psi[4*j - 1] = torch.log(torch.clamp(theta[4*j - 1] - theta[4*j - 2], min=EPS))
            psi[4*j + 0] = torch.log(torch.clamp(theta[4*j + 0] - theta[4*j - 1], min=EPS))
            psi[4*j + 1] = torch.log(torch.clamp(theta[4*j + 1] - theta[4*j + 0], min=EPS))

    psi[n - 2] = torch.log(torch.clamp(theta[n - 3] - theta[n - 2], min=EPS))
    psi[n - 1] = torch.log(torch.clamp(theta[n - 1] - theta[n - 2], min=EPS))

    return psi


# ======================================================
# CASPs-Free (calculate_θ_casps_free)
# no inter-MF ordering constraints
# ======================================================

def theta_from_psi_free(psi: torch.Tensor, normalised: bool = False) -> torch.Tensor:
    psi = psi.flatten()
    n = psi.numel()
    theta = torch.zeros_like(psi)

    theta[0] = psi[0]
    theta[1] = theta[0] + torch.exp(psi[1])

    if n > 4:
        J = n // 4 - 1
        for j in range(1, J + 1):
            theta[4*j - 2] = psi[4*j - 2]
            theta[4*j - 1] = theta[4*j - 2] + torch.exp(psi[4*j - 1])
            theta[4*j + 0] = theta[4*j - 1] + torch.exp(psi[4*j + 0])
            theta[4*j + 1] = theta[4*j + 0] + torch.exp(psi[4*j + 1])

    theta[n - 2] = psi[n - 2]
    theta[n - 1] = theta[n - 2] + torch.exp(psi[n - 1])

    if normalised:
        theta = torch.sigmoid(theta)

    return theta


def psi_from_theta_free(theta: torch.Tensor, normalised: bool = False) -> torch.Tensor:
    theta = theta.flatten()
    n = theta.numel()
    psi = torch.zeros_like(theta)

    if normalised:
        theta = torch.log(theta / (1 - theta))

    psi[0] = theta[0]
    psi[1] = torch.log(torch.clamp(theta[1] - theta[0], min=EPS))

    if n > 4:
        J = n // 4 - 1
        for j in range(1, J + 1):
            psi[4*j - 2] = theta[4*j - 2]
            psi[4*j - 1] = torch.log(torch.clamp(theta[4*j - 1] - theta[4*j - 2], min=EPS))
            psi[4*j + 0] = torch.log(torch.clamp(theta[4*j + 0] - theta[4*j - 1], min=EPS))
            psi[4*j + 1] = torch.log(torch.clamp(theta[4*j + 1] - theta[4*j + 0], min=EPS))

    psi[n - 2] = theta[n - 2]
    psi[n - 1] = torch.log(torch.clamp(theta[n - 1] - theta[n - 2], min=EPS))

    return psi


# ======================================================
# θ -> trapezoidal MF decoding (shared, EXACT)
# ======================================================

def decode_trapmf(theta: torch.Tensor, m: int, j: int):
    n = theta.numel()

    if j == 1:
        return -INF, -INF, theta[0], theta[1]

    if j == m:
        return theta[n - 2], theta[n - 1], INF, INF

    start = (j - 1) * 4 - 2
    return theta[start], theta[start + 1], theta[start + 2], theta[start + 3]
