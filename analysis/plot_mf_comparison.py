"""
Generate qualitative MF comparison plots for a representative dataset.

This script visualises the input membership functions (MFs) of a fuzzy
inference system before optimisation and after training with different
optimisation methods (e.g. Adam vs PSO).

The resulting figure is intended for inclusion in the Results section
as qualitative evidence of optimisation behaviour.
"""

import os
import sys
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt

# ============================================================
# Add project root to sys.path
# ============================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ============================================================
# Project imports
# ============================================================

from experiments.utils.datasets import load_dataset
from experiments.experiment_config import CONFIG
from experiments.initialisation_registry import build_fis_from_config
from experiments.optimiser_registry import OPTIMISERS
from src.trainable_fis import TrainableFIS


# ============================================================
# Settings
# ============================================================

DATASET = "Energy"              # representative dataset
OPTIMISERS_TO_COMPARE = ["adam", "pso"]  # gradient vs non-gradient
SEED = 0
POINT_N = 201
SAVE_PATH = "analysis/mf_comparison_energy.png"


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Load dataset
# ============================================================

X_train, X_test, y_train, y_test = load_dataset(DATASET)
n_inputs = X_train.shape[1]


# ============================================================
# Helper: train model and return FIS
# ============================================================

def train_fis(opt_method, base_cfg):
    cfg = copy.deepcopy(base_cfg)
    cfg["optimiser"]["method"] = opt_method

    fis = build_fis_from_config(X_train, y_train, cfg["initialisation"])
    model = TrainableFIS(fis)

    model, _ = OPTIMISERS[opt_method](model, X_train, y_train, cfg)
    return model.fis


# ============================================================
# Build initial and trained FIS
# ============================================================

base_cfg = copy.deepcopy(CONFIG)

# Initial (untrained)
fis_initial = build_fis_from_config(X_train, y_train, base_cfg["initialisation"])

# Trained
trained_fis = {}
for opt in OPTIMISERS_TO_COMPARE:
    trained_fis[opt] = train_fis(opt, base_cfg)


# ============================================================
# Plot MF comparison
# ============================================================

rows = 1 + len(OPTIMISERS_TO_COMPARE)   # Initial + trained
cols = n_inputs

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(4 * cols, 3 * rows),
    squeeze=False
)

row_labels = ["Initial"] + [opt.upper() for opt in OPTIMISERS_TO_COMPARE]

for r in range(rows):
    for c in range(cols):
        ax = axes[r, c]

        if r == 0:
            fis = fis_initial
        else:
            fis = trained_fis[OPTIMISERS_TO_COMPARE[r - 1]]

        # ----- replicate plotmf logic, but draw on ax -----
        var = fis.input[c]
        r0, r1 = var["range"]
        x = np.linspace(r0, r1, POINT_N)

        for mf in var["mf"]:
            MF = fis.evalmf(
                torch.tensor(x, dtype=torch.float32),
                mf["type"],
                mf["params"]
            ).detach().cpu().numpy()

            ax.plot(x, MF, linewidth=2)

        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)

        if r == 0:
            ax.set_title(f"Input {c+1}", fontsize=12)
        if c == 0:
            ax.set_ylabel(row_labels[r], fontsize=12)

fig.suptitle(
    f"Optimised Input Membership Functions ({DATASET} dataset)",
    fontsize=14,
    y=1.02
)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"[Saved] MF comparison figure → {SAVE_PATH}")
