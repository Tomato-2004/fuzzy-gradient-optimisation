import torch
from src.FuzzyInferenceSystem import FuzzyInferenceSystem

def build_basic_fis(num_inputs):
    """
    Build a TS-type FIS:
        - Each input has 3 Gaussian MFs
        - Output is TS linear (no output MFs)
        - Rules use TS format: {antecedent: [...], coeff: [...]}
    """

    # -----------------------
    # IMPORTANT: TS MODE
    # -----------------------
    fis = FuzzyInferenceSystem("baseline", fis_type="ts")

    theta_list = []  # trainable parameters

    # -----------------------
    # 1. Create inputs + MFs
    # -----------------------
    for i in range(num_inputs):
        fis.add_variable("input", f"x{i+1}", (0, 1))

        mf_params = [
            torch.tensor([0.1, 0.2], requires_grad=True),
            torch.tensor([0.4, 0.5], requires_grad=True),
            torch.tensor([0.7, 0.8], requires_grad=True),
        ]

        for j, params in enumerate(mf_params):
            fis.add_mf("input", i, f"mf{j + 1}", "gaussmf", params.detach().numpy())
            theta_list.append(params)

    # -----------------------
    # 2. Build TS rules
    #    Example: 3 MFs per input → choose MF 0 for all inputs
    # -----------------------

    rules = []
    for r in range(3):  
        # create 3 TS rules (each using different MF index)
        antecedent = [r % 3] * num_inputs  # each input picks MF index r
        coeff = torch.zeros(num_inputs + 1, requires_grad=True)  # b0 + bi*x
        coeff.data[:] = torch.randn_like(coeff)

        rules.append({
            "antecedent": antecedent,
            "coeff": coeff
        })

        theta_list.append(coeff)

    # register rules in the FIS
    fis.rule = rules

    return fis, theta_list
