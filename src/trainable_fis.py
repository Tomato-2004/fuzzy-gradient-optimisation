import torch
from src.FuzzyInferenceSystem import FuzzyInferenceSystem


class TrainableFIS(torch.nn.Module):
    """
    Wrapper to make FIS trainable with PyTorch.
    theta_list 通常用来收集需要训练的参数（比如 TS 里的线性后件 coef）。
    这里保持原来的接口和名字不变。
    """

    def __init__(self, fis: FuzzyInferenceSystem, theta_list):
        super().__init__()
        self.fis = fis

        # 保留你的 ParameterList 结构，但不强行在前向里用它
        # （真正参与计算的参数目前主要是 FIS.rule 里挂的 coeff）
        self.theta = torch.nn.ParameterList([
            torch.nn.Parameter(t.clone().detach().float())
            for t in theta_list
        ])

    def forward(self, X):
        """
        对外接口保持不变：接收 X，返回一个 1D tensor (batch_size,)

        - 如果 FIS 是 TS / CASP 类型：调用 self.fis.eval_ts(X)，保留梯度
        - 否则（Mamdani 等）：走原来的逐行 eval() + defuzz 流程（不可微）
        """

        # ------------------------------
        # 1. TS / CASP：使用可微分的 eval_ts
        # ------------------------------
        if hasattr(self.fis, "type") and isinstance(self.fis.type, str) \
                and self.fis.type.lower() in ["ts", "casp"]:

            # 确保是 tensor
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)

            # eval_ts 本身支持 batch 输入，直接调用
            y = self.fis.eval_ts(X)   # 返回 (batch,) 或 (batch,1) 形状的 tensor

            # 保证是 1D float32 tensor
            return y.float().view(-1)

        # ------------------------------
        # 2. 其他类型（Mamdani）：沿用原有逻辑（不可微分）
        # ------------------------------
        outputs = []

        # 允许 X 是 numpy array 或 torch tensor
        if isinstance(X, torch.Tensor):
            rows = X
        else:
            # 假定是 numpy 数组或 list
            rows = X

        for row in rows:
            if isinstance(row, torch.Tensor):
                row_in = row.detach().cpu().numpy().tolist()
            else:
                # numpy 数组或者 list
                row_in = row.tolist()

            # 对于 Mamdani，eval 返回的是普通 Python float
            y = self.fis.eval(row_in)
            outputs.append(float(y))

        return torch.tensor(outputs, dtype=torch.float32)
