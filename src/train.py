"""
Training script for the PowerPredict ANN model.
"""
import argparse
import os
import numpy as np
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import load_and_preprocess_data, save_scaler
from src.model import ANN


def train(
    epochs: int = 100,
    batch_size: int = 32,
    data_path: str = None,
    output_dir: str = "models",
    random_state: int = 41
):
    """Execute training pipeline, evaluate metrics, and save best model & scaler."""
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "best_model.pt")
    scaler_path = os.path.join(output_dir, "scaler.joblib")

    print(f"Loading data and preparing DataLoaders (batch_size={batch_size})...")
    (
        train_loader,
        test_loader,
        X_train_tensor,
        y_train_tensor,
        X_test_tensor,
        y_test_tensor,
        scaler
    ) = load_and_preprocess_data(
        data_path=data_path,
        batch_size=batch_size,
        random_state=random_state
    )

    # Save the fitted scaler
    save_scaler(scaler, scaler_path)
    print(f"Saved fitted scaler to {scaler_path}")

    # Initialize model, loss, and optimizer
    model = ANN(input_dim=4)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                outputs = model(xb)
                loss = criterion(outputs, yb)
                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(test_loader)
        val_losses.append(epoch_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {epoch_train_loss:10.4f} | Val Loss: {epoch_val_loss:10.4f}")

        # Checkpointing best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)

    print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")
    print(f"Saved best model checkpoint to {best_model_path}")

    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        train_preds = model(X_train_tensor)
        test_preds = model(X_test_tensor)

        train_mse = criterion(train_preds, y_train_tensor).item()
        test_mse = criterion(test_preds, y_test_tensor).item()
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test_tensor.numpy(), test_preds.numpy())

    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS (Best Model Checkpoint)")
    print("=" * 50)
    print(f"  Training MSE:   {train_mse:8.4f}")
    print(f"  Testing MSE:    {test_mse:8.4f}")
    print(f"  Testing RMSE:   {test_rmse:8.4f} MW")
    print(f"  Testing R²:     {test_r2:8.4f} ({test_r2*100:.2f}%)")
    print("=" * 50 + "\n")

    return {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "best_val_loss": best_val_loss
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PowerPredict ANN Regression Model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--data-path", type=str, default=None, help="Path to powerplant_data.csv")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model & scaler")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
