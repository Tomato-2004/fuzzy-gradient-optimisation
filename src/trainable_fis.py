# src/trainable_fis.py
import torch
from torch import nn
from .FuzzyInferenceSystem import FuzzyInferenceSystem


class TrainableFIS(nn.Module):
    """
    把 FuzzyInferenceSystem 包装成 nn.Module，管理可学习参数（MF params）。

    当前版本：
        - 所有输入/输出 MF 的参数都是 nn.Parameter
        - 规则结构固定，不学习
        - forward 使用 eval_batch (快速批量推理)
        - 核心推理逻辑仍然全部在 FuzzyInferenceSystem
    """

    def __init__(self, fis: FuzzyInferenceSystem):
        super().__init__()
        self.fis = fis

        params = []

        # 把 FIS 里所有 mf["params"] 都替换成 nn.Parameter
        for var_list in (self.fis.input, self.fis.output):
            for var in var_list:
                for mf in var["mf"]:
                    # 注意：mf["params"] 可能已经是 tensor
                    p = nn.Parameter(
                        torch.as_tensor(mf["params"], dtype=torch.float32)
                    )
                    mf["params"] = p          # 替换回 FIS 内部
                    params.append(p)

        # 统一管理
        self.mf_params = nn.ParameterList(params)

    def forward(self, x: torch.Tensor, point_n: int = 101) -> torch.Tensor:
        """
        x: (batch_size, n_inputs) 或 (n_inputs,)
        返回: (batch_size,) tensor
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 使用快速 vectorized 推理
        return self.fis.eval_batch(x, point_n=point_n)
