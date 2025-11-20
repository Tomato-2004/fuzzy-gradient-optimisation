import os
import sys

sys.path.append(os.path.abspath("."))

from experiments.utils.data_loader import load_dataset
from experiments.utils.evaluator import rmse
from experiments.utils.fis_builder import build_basic_fis
from experiments.train_adam import train_with_adam

from src.trainable_fis import TrainableFIS


def run_experiment_basic():

    dataset = "Airfoil_self_noise"
    dataset_path = f"data/datasets/{dataset}.csv"

    print("=======================================")
    print(f" Running BASIC FIS experiment on: {dataset}")
    print("=======================================")

    # -----------------------------
    # Load dataset
    # -----------------------------
    X_train, y_train, X_test, y_test = load_dataset(dataset_path)
    print("[OK] Dataset loaded")

    # -----------------------------
    # Build simple FIS
    # -----------------------------
    fis, theta_list = build_basic_fis(num_inputs=X_train.shape[1])
    model = TrainableFIS(fis, theta_list)

    # -----------------------------
    # Train using ADAM
    # -----------------------------
    trained_model = train_with_adam(
        model,
        X_train,
        y_train,
        lr=0.01,
        epochs=30
    )

    # -----------------------------
    # Evaluate
    # -----------------------------
    y_pred_train = trained_model.forward(X_train).detach()
    y_pred_test = trained_model.forward(X_test).detach()

    print("\n===== RESULTS =====")
    print(f"Train RMSE = {rmse(y_train, y_pred_train):.4f}")
    print(f"Test  RMSE = {rmse(y_test, y_pred_test):.4f}")
    print("===========================")


if __name__ == "__main__":
    run_experiment_basic()
