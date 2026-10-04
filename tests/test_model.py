"""
Unit tests for the PowerPredict model and dataset pipeline.
"""
import pytest
import torch
from src.model import ANN
from src.dataset import find_data_file, load_and_preprocess_data


def test_model_architecture():
    """Verify ANN layer setup and exact parameter count."""
    model = ANN(input_dim=4)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 4*6 + 6 (layer 1) + 6*6 + 6 (layer 2) + 6*1 + 1 (output) = 30 + 42 + 7 = 79
    assert total_params == 79, f"Expected 79 parameters, found {total_params}"


def test_model_forward_pass_shapes():
    """Verify input-output tensor shapes for various batch sizes."""
    model = ANN(input_dim=4)
    model.eval()

    batch_sizes = [1, 16, 32, 64]
    for b in batch_sizes:
        dummy_input = torch.randn(b, 4)
        output = model(dummy_input)
        assert output.shape == (b, 1), f"Expected shape ({b}, 1), got {output.shape}"


def test_data_loader_pipeline():
    """Verify data loading and mini-batch tensor properties."""
    data_file = find_data_file()
    assert data_file is not None

    train_loader, test_loader, X_train_t, y_train_t, X_test_t, y_test_t, scaler = load_and_preprocess_data(
        data_path=data_file,
        batch_size=32,
        random_state=41
    )

    # Check dataset split sizes (9568 total: 7654 train, 1914 test)
    assert len(X_train_t) == 7654
    assert len(X_test_t) == 1914

    # Check mini-batch shape
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (32, 4)
    assert batch_y.shape == (32, 1)


def test_prediction_output_range():
    """Verify model produces reasonable continuous outputs."""
    model = ANN(input_dim=4)
    model.eval()
    sample = torch.tensor([[0.5, -0.2, 1.1, -0.8]], dtype=torch.float32)
    output = model(sample)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_power_predictor_inference():
    """Verify end-to-end inference wrapper produces realistic MW values."""
    from src.predict import PowerPredictor
    import os
    if os.path.exists("models/best_model.pt") and os.path.exists("models/scaler.joblib"):
        predictor = PowerPredictor()
        # Ambient temp 15°C, Vacuum 40 cm Hg, Pressure 1013 mbar, Humidity 70%
        pred = predictor.predict([[15.0, 40.0, 1013.0, 70.0]])
        # Expected to be well within plant operating range [420, 496] MW
        assert 420.0 <= float(pred) <= 496.0
