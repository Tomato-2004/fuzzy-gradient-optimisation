import torch
import numpy as np


def train_with_rmsprop(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    num_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    alpha: float = 0.99,
    weight_decay: float = 0.0,
    point_n: int = 101,
    device: str = "cpu",
):
    """
    RMSProp optimiser for fuzzy systems.
    """

    model.to(device)
    model.train()

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=lr,
        alpha=alpha,
        weight_decay=weight_decay,
    )

    X = torch.tensor(X_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)

    n_samples = X.shape[0]
    history = []

    for epoch in range(num_epochs):
        perm = torch.randperm(n_samples, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            xb = X[idx]
            yb = y[idx]

            optimizer.zero_grad()
            preds = model(xb, point_n=point_n)  
            loss = torch.mean((preds - yb) ** 2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= n_batches
        history.append(epoch_loss)

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(
                f"[Epoch {epoch+1:4d}/{num_epochs}] "
                f"train_loss={epoch_loss:.6f}"
            )

    return model, history
