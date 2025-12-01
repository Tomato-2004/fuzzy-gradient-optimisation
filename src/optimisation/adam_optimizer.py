# src/optimisation/adam_optimizer.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # 仅用于类型提示，运行时不会真正导入，避免循环依赖
    from ..trainable_fis import TrainableFIS


def create_dataloader(X, y, batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    把 (X, y) 包装成 DataLoader，统一 tensor 类型 & 形状。
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)

    if y.dim() == 1:
        y = y.unsqueeze(1)  # (N,) -> (N, 1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def train_with_adam(
    trainable_fis: "TrainableFIS",
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    num_epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    point_n: int = 101,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = torch.nn.MSELoss(),
    adam_betas=(0.9, 0.999),
    adam_eps: float = 1e-8,
    device: str = "cpu",
    callbacks: Optional[Dict[str, Callable[[Dict[str, Any]], None]]] = None,
):
    """
    用 Adam 训练一个 TrainableFIS（即你的 Mamdani FIS + 可学习 MF 参数）。

    说明：
        - 保持接口通用，后续可以复用这个 train_with_* 接口给 GA/PSO 等非梯度方法。
        - point_n 传给 fis.eval，用于 defuzz 时的输出 universo 采样点数。
    """
    callbacks = callbacks or {}
    model = trainable_fis.to(device)

    train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)

    if X_val is not None and y_val is not None:
        val_loader = create_dataloader(X_val, y_val, batch_size=len(X_val), shuffle=False)
    else:
        val_loader = None

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=adam_betas,
        eps=adam_eps,
        weight_decay=weight_decay,
    )

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            y_pred = model(xb, point_n=point_n)
            if y_pred.dim() == 1:
                y_pred = y_pred.unsqueeze(1)

            loss = loss_fn(y_pred, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)

        avg_train_loss = running_loss / n_samples
        history["train_loss"].append(avg_train_loss)

        # 验证集
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                v_loss = 0.0
                v_n = 0
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    y_pred = model(xb, point_n=point_n)
                    if y_pred.dim() == 1:
                        y_pred = y_pred.unsqueeze(1)
                    loss = loss_fn(y_pred, yb)
                    v_loss += loss.item() * xb.size(0)
                    v_n += xb.size(0)
                avg_val_loss: Optional[float] = v_loss / v_n
        else:
            avg_val_loss = None

        history["val_loss"].append(avg_val_loss)

        # 简单日志
        if epoch % max(1, num_epochs // 10) == 0 or epoch == 1:
            if avg_val_loss is not None:
                print(f"[Epoch {epoch:4d}/{num_epochs}] "
                      f"train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")
            else:
                print(f"[Epoch {epoch:4d}/{num_epochs}] "
                      f"train_loss={avg_train_loss:.6f}")

        # 回调钩子（例如早停 / 可视化）
        if "on_epoch_end" in callbacks:
            callbacks["on_epoch_end"](
                {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "model": model,
                    "optimizer": optimizer,
                    "history": history,
                }
            )

    return model, history
