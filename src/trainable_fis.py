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
        - forward 支持 batch 输入
    """

    def __init__(self, fis: FuzzyInferenceSystem):
        super().__init__()
        self.fis = fis

        params = []

        for var_list in (self.fis.input, self.fis.output):
            for var in var_list:
                for mf in var["mf"]:
                    p = torch.nn.Parameter(
                        torch.tensor(mf["params"], dtype=torch.float32)
                    )
                    mf["params"] = p     # 替换回 FIS 内部
                    params.append(p)

        self.mf_params = nn.ParameterList(params)

        # 预留：以后可以加 CASP 的 ψ 参数并在 forward 里做 θ(ψ) 映射

    def forward(self, x: torch.Tensor, point_n: int = 101) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (n,) -> (1, n)

        outputs = []
        for i in range(x.shape[0]):
            y = self.fis.eval(x[i], point_n=point_n)  # scalar tensor
            outputs.append(y)

        return torch.stack(outputs, dim=0)  # (batch,)
