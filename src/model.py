"""
Neural Network Architecture for CCPP Electrical Power Output Prediction.
"""
import torch
import torch.nn as nn


class ANN(nn.Module):
    """
    Feedforward Artificial Neural Network (Multilayer Perceptron) for regression.
    
    Architecture:
        Input Layer (4 features: AT, V, AP, RH)
        Hidden Layer 1: Linear(4, 6) + ReLU
        Hidden Layer 2: Linear(6, 6) + ReLU
        Output Layer:   Linear(6, 1) -> Net Hourly Electrical Energy Output (PE)
    """
    def __init__(self, input_dim: int = 4):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            # 1st hidden layer
            nn.Linear(input_dim, 6),
            nn.ReLU(),
            # 2nd hidden layer
            nn.Linear(6, 6),
            nn.ReLU(),
            # Output layer (linear activation for regression)
            nn.Linear(6, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass propagation."""
        return self.model(x)
