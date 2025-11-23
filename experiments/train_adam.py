# experiments/train_adam.py
import os
import sys
import numpy as np
import torch

# 让 Python 找到 src
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.rule_generator.genfis import build_initial_fis_from_data
from src.rule_generator.genrule import genrules_kmeans
from src.trainable_fis import TrainableFIS
from src.optimisation.adam_optimizer import train_with_adam
from experiments.utils.datasets import load_dataset


def train_one_dataset(
    dataset_name: str,
    n_mfs_per_input: int = 3,
    n_mfs_output: int = 3,
    n_rules: int = 10,
    num_epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    point_n: int = 101,
    random_state: int = 0,
):
    print(f"\n===== Training dataset: {dataset_name} =====")

    X_train, X_test, y_train, y_test = load_dataset(dataset_name, test_size=0.2,
                                                    random_state=random_state)

    # 1. 用数据构造初始 FIS（genfis）
    fis = build_initial_fis_from_data(
        X_train,
        y_train,
        n_mfs_per_input=n_mfs_per_input,
        n_mfs_output=n_mfs_output,
        fis_name=f"mamdani_{dataset_name}",
    )

    # 2. 用 k-means 生成规则（genrule）
    genrules_kmeans(
        fis,
        X_train,
        y_train,
        n_rules=n_rules,
        weight=1.0,
        and_or=1,
        random_state=random_state,
    )

    fis.summary()

    # 3. 包装成可训练模型
    model = TrainableFIS(fis)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4. 用 Adam 训练
    trained_model, history = train_with_adam(
        model,
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        point_n=point_n,
        device=device,
    )

    # 5. 计算最终 MSE（train / test）
    model.eval()
    with torch.no_grad():
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

        y_train_pred = trained_model(X_train_t, point_n=point_n).cpu().numpy()
        y_test_pred = trained_model(X_test_t, point_n=point_n).cpu().numpy()

        train_mse = float(np.mean((y_train_pred.reshape(-1) - y_train) ** 2))
        test_mse = float(np.mean((y_test_pred.reshape(-1) - y_test) ** 2))

    print(f"[{dataset_name}] Final train MSE = {train_mse:.4f}, "
          f"test MSE = {test_mse:.4f}")

    return {
        "dataset": dataset_name,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "history": history,
    }


if __name__ == "__main__":
    # 先跑一个数据集试试，例如 Abalone
    result = train_one_dataset(
    "Abalone",
    n_mfs_per_input=2,   # ↓ 输入 MF 少一点
    n_mfs_output=2,      # ↓ 输出 MF 少一点
    n_rules=5,           # ↓ 规则数少一点
    num_epochs=20,       # ↓ epoch 少一点
    lr=5e-3,             # ↑ 提高学习率（更快收敛）
    batch_size=16,       # ↓ batch 更小（更快）
    point_n=31,          # ↓ defuzz 点数减少
    )

    print(result["dataset"], "done.")
