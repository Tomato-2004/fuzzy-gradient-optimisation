# src/trainable_fis.py
import numpy as np
import torch
from torch import nn
from .FuzzyInferenceSystem import FuzzyInferenceSystem


class TrainableFIS(nn.Module):
    """
    把 FuzzyInferenceSystem 包装成 nn.Module，管理可学习参数（MF params）。

    特点：
        - 所有输入/输出 MF 参数都是 nn.Parameter
        - 规则结构固定，不学习
        - forward 使用 eval_batch（快速批量推理）
        - 新增 flatten_params / assign_params，使多种优化器可统一调用
    """

    def __init__(self, fis: FuzzyInferenceSystem):
        super().__init__()
        self.fis = fis

        params = []

        # 把 FIS 里所有 mf["params"] 替换为 nn.Parameter
        for var_list in (self.fis.input, self.fis.output):
            for var in var_list:
                for mf in var["mf"]:
                    p = nn.Parameter(torch.as_tensor(mf["params"], dtype=torch.float32))
                    mf["params"] = p
                    params.append(p)

        # 统一管理所有 MF 参数
        self.mf_params = nn.ParameterList(params)

    # ============================================================
    #     通用参数接口：flatten_params() / assign_params()
    # ============================================================

    def flatten_params(self):
        """
        提取所有 MF 参数为一个 numpy 向量（供 PSO / GA / CMA-ES 等使用）
        """
        vec = []
        for p in self.mf_params:
            vec.append(p.data.detach().cpu().numpy().reshape(-1))
        return np.concatenate(vec)

    def assign_params(self, vec):
        """
        把一个向量（来自 PSO / GA / Random Search）写回 FIS 的所有 MF 参数
        """
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
