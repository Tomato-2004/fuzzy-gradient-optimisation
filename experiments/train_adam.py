import torch

def train_with_adam(model, X_train, y_train, lr=0.01, epochs=20):

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t  = torch.tensor(y_train,  dtype=torch.float32)

    # ============================================================
    # 识别 FIS 类型（兼容 TrainableFIS）
    # ============================================================
    fis_type = None
    if hasattr(model, "fis") and hasattr(model.fis, "type"):
        fis_type = model.fis.type.lower()

    # ============================================================
    # TS / CASP 模型：训练 rule["coeff"]
    # ============================================================
    if fis_type in ["ts", "casp"]:
        params = []
        for rule in model.fis.rule:
            coeff = rule["coeff"]
            if not isinstance(coeff, torch.nn.Parameter):
                rule["coeff"] = torch.nn.Parameter(
                    torch.tensor(coeff, dtype=torch.float32),
                    requires_grad=True
                )
            params.append(rule["coeff"])

        optimizer = torch.optim.Adam(params, lr=lr)

        def forward_fn(x):
            return model.fis.eval_ts(x)

    # ============================================================
    # Mamdani 模型：使用 TrainableFIS 的参数
    # ============================================================
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        def forward_fn(x):
            return model.forward(x)

    # ============================================================
    # Training loop
    # ============================================================
    for epoch in range(epochs):

        optimizer.zero_grad()
        y_pred = forward_fn(X_train_t)

        loss = torch.mean((y_pred - y_train_t) ** 2)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"[Epoch {epoch}] Loss = {loss.item():.4f}")

    return model
