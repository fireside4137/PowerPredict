"""
Data loading and preprocessing pipeline for PowerPredict.
"""
import os
from typing import Tuple
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

FEATURE_COLUMNS = ["AT", "V", "AP", "RH"]
TARGET_COLUMN = "PE"


def find_data_file(custom_path: str = None) -> str:
    """Locate the powerplant_data.csv file across common project locations."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    candidates = [
        os.path.join("data", "powerplant_data.csv"),
        "powerplant_data.csv",
        os.path.join("..", "data", "powerplant_data.csv"),
        os.path.join("notebooks", "powerplant_data.csv")
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find powerplant_data.csv in any expected locations: {candidates}")


def load_and_preprocess_data(
    data_path: str = None,
    test_size: float = 0.2,
    random_state: int = 41,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
    """
    Load dataset, perform train/test split, standardize features, and create PyTorch DataLoaders.
    
    Returns:
        train_loader, test_loader, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler
    """
    file_path = find_data_file(data_path)
    df = pd.read_csv(file_path)

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    # Train-test split (80-20 split with exact seed 41)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Build TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        test_loader,
        X_train_tensor,
        y_train_tensor,
        X_test_tensor,
        y_test_tensor,
        scaler
    )


def save_scaler(scaler: StandardScaler, file_path: str = os.path.join("models", "scaler.joblib")) -> None:
    """Save fitted StandardScaler object using joblib."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(scaler, file_path)


def load_scaler(file_path: str = os.path.join("models", "scaler.joblib")) -> StandardScaler:
    """Load fitted StandardScaler object."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Scaler file not found at: {file_path}")
    return joblib.load(file_path)
