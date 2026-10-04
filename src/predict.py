"""
Inference module for PowerPredict.
Predicts net hourly electrical energy output (PE in MW) given ambient thermodynamic variables.
"""
import argparse
import os
from typing import Union, List
import numpy as np
import pandas as pd
import torch

from src.dataset import load_scaler, FEATURE_COLUMNS
from src.model import ANN


class PowerPredictor:
    """Wrapper class for loading trained ANN weights & scaler and making predictions."""
    def __init__(
        self,
        model_path: str = os.path.join("models", "best_model.pt"),
        scaler_path: str = os.path.join("models", "scaler.joblib")
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Run training first.")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Run training first.")

        self.scaler = load_scaler(scaler_path)
        self.model = ANN(input_dim=4)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def predict(self, input_data: Union[np.ndarray, pd.DataFrame, List[List[float]]]) -> np.ndarray:
        """
        Make electrical energy output predictions for one or more observations.
        
        Args:
            input_data: 2D array-like structure with columns [AT, V, AP, RH].
            
        Returns:
            np.ndarray: Predicted electrical power output in Megawatts (MW).
        """
        if isinstance(input_data, pd.DataFrame):
            df_input = input_data[FEATURE_COLUMNS]
        else:
            arr = np.array(input_data, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            df_input = pd.DataFrame(arr, columns=FEATURE_COLUMNS)

        # Scale features
        scaled_features = self.scaler.transform(df_input)
        features_tensor = torch.tensor(scaled_features, dtype=torch.float32)

        with torch.no_grad():
            preds = self.model(features_tensor)

        return preds.squeeze().numpy()


def main():
    parser = argparse.ArgumentParser(description="Predict Electrical Energy Output (MW)")
    parser.add_argument("--at", type=float, default=15.0, help="Ambient Temperature (°C)")
    parser.add_argument("--v", type=float, default=40.0, help="Exhaust Vacuum (cm Hg)")
    parser.add_argument("--ap", type=float, default=1013.0, help="Ambient Pressure (mbar)")
    parser.add_argument("--rh", type=float, default=70.0, help="Relative Humidity (%)")
    parser.add_argument("--file", type=str, default=None, help="Path to CSV file with features [AT, V, AP, RH]")
    parser.add_argument("--model-path", type=str, default=os.path.join("models", "best_model.pt"))
    parser.add_argument("--scaler-path", type=str, default=os.path.join("models", "scaler.joblib"))

    args = parser.parse_args()

    predictor = PowerPredictor(model_path=args.model_path, scaler_path=args.scaler_path)

    if args.file:
        df = pd.read_csv(args.file)
        preds = predictor.predict(df)
        df["Predicted_PE_MW"] = preds
        output_file = "predictions.csv"
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file} ({len(preds)} rows)")
    else:
        features = [[args.at, args.v, args.ap, args.rh]]
        pred = predictor.predict(features)
        print("\n" + "=" * 45)
        print("POWER OUTPUT PREDICTION")
        print("=" * 45)
        print(f"  Ambient Temperature (AT):  {args.at:8.2f} °C")
        print(f"  Exhaust Vacuum (V):        {args.v:8.2f} cm Hg")
        print(f"  Ambient Pressure (AP):     {args.ap:8.2f} mbar")
        print(f"  Relative Humidity (RH):    {args.rh:8.2f} %")
        print("-" * 45)
        print(f"  Predicted Power Output:    {float(pred):8.2f} MW")
        print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
