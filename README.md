# PowerPredict: Combined Cycle Power Plant Energy Output Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A deep learning regression system developed in **PyTorch** to predict the net hourly electrical energy output (**PE**, in Megawatts) of a **Combined Cycle Power Plant (CCPP)** using ambient thermodynamic variables.

---

## 📌 Project Overview

In a Combined Cycle Power Plant, electricity is produced through a combination of gas turbines (Brayton cycle) and steam turbines (Rankine cycle). Thermodynamic efficiency and power output are heavily governed by ambient conditions:
- **Temperature ($AT$)**: Warmer air is less dense, reducing mass flow rate through the compressor and gas turbine power output.
- **Vacuum ($V$)**: Higher exhaust condenser vacuum pressure impedes the expansion ratio of the steam turbine.
- **Ambient Pressure ($AP$)**: Influences air density and intake compressor efficiency.
- **Relative Humidity ($RH$)**: Alters air density and heat capacity, affecting combustion thermodynamics.

This project implements a multi-layer **Artificial Neural Network (ANN)** to accurately model these non-linear physical interactions.

---

## 📊 Dataset Description

The dataset consists of **9,568 hourly observations** collected from a Combined Cycle Power Plant operating at base load over a 6-year period.

| Feature | Name | Units | Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`AT`** | Ambient Temperature | °C | 1.81 – 37.11 | Air temperature entering gas turbine |
| **`V`** | Exhaust Vacuum | cm Hg | 25.36 – 81.56 | Steam turbine exhaust pressure |
| **`AP`** | Ambient Pressure | mbar | 992.89 – 1033.30 | Barometric ambient pressure |
| **`RH`** | Relative Humidity | % | 25.56 – 100.16 | Atmospheric humidity level |
| **`PE`** | Net Electrical Energy Output | **MW** | **420.26 – 495.76** | **Target variable** to predict |

---

## 🧠 Neural Network Architecture

The predictive model is a Multilayer Perceptron (MLP) implemented in PyTorch (`src/model.py`):

```
Input Features (4) 
    │
    ▼
Linear (4 → 6) + ReLU
    │
    ▼
Linear (6 → 6) + ReLU
    │
    ▼
Linear (6 → 1) ──► Predicted Power Output (PE in MW)
```

- **Input Dimension:** 4 features (`AT`, `V`, `AP`, `RH`)
- **Hidden Layers:** 2 layers with 6 units each, activated by **ReLU**
- **Output Layer:** 1 continuous linear node
- **Total Trainable Parameters:** 79
- **Loss Function:** Mean Squared Error ($\text{MSELoss}$)
- **Optimizer:** Adam Optimizer ($\alpha = 0.001$)
- **Mini-batch Size:** 32 with dynamic shuffling

---

## 📈 Performance & Evaluation

The model was evaluated against an unseen **20% test set (1,914 samples)**:

| Metric | Training Set | Testing Set |
| :--- | :---: | :---: |
| **Mean Squared Error (MSE)** | **20.45** | **20.20** |
| **Root Mean Squared Error (RMSE)** | **4.52 MW** | **4.49 MW** |
| **Coefficient of Determination ($R^2$)** | — | **0.9276 (~92.8%)** |

### Visualizations

#### 1. Training & Validation Loss Minimization
Loss converges rapidly within 25 epochs and remains stable through 100 epochs with zero signs of overfitting:
<p align="center">
  <img src="assets/Minimization%20of%20Loss.png" width="750" alt="Minimization of Loss">
</p>

#### 2. Actual vs. Predicted Power Output
Predictions cluster tightly along the identity ($y = x$) reference trajectory across the full power output range (420–495 MW):
<p align="center">
  <img src="assets/Actual%20vs%20predicted.png" width="750" alt="Actual vs Predicted">
</p>

#### 3. Residuals & Error Distribution
Residual errors ($y - \hat{y}$) are symmetrically distributed and centered at zero error:
<p align="center">
  <img src="assets/Distribution%20of%20residuals.png" width="750" alt="Distribution of Residuals">
</p>

---

## 📁 Repository Structure

```text
PowerPredict/
├── assets/
│   ├── Actual vs predicted.png         # Regression fit scatter plot
│   ├── Distribution of residuals.png    # Error distribution histogram
│   └── Minimization of Loss.png         # Training vs validation loss curve
├── data/
│   └── powerplant_data.csv             # Combined Cycle Power Plant dataset
├── models/
│   ├── best_model.pt                   # Checkpointed PyTorch model weights
│   └── scaler.joblib                   # Fitted StandardScaler object
├── notebooks/
│   ├── ANN_regression.ipynb            # Original exploratory Jupyter notebook
│   └── powerplant_data.csv             # Backwards-compatible notebook dataset
├── src/
│   ├── __init__.py                     # Package initialization
│   ├── model.py                        # PyTorch ANN class definition
│   ├── dataset.py                      # Data pipeline, scaling & DataLoaders
│   ├── train.py                        # Training, validation & checkpointing script
│   └── predict.py                      # Single & batch inference CLI utility
├── tests/
│   ├── __init__.py
│   └── test_model.py                   # Automated unit tests
├── .gitignore                          # Git ignore definitions
├── LICENSE                             # MIT License
├── README.md                           # Documentation
└── requirements.txt                    # Project dependencies
```

---

## 🚀 Quick Start

### 1. Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/your-username/PowerPredict.git
cd PowerPredict
pip install -r requirements.txt
```

### 2. Training the Model

Train the model and save the best checkpoint to `models/best_model.pt`:

```bash
python -m src.train --epochs 100 --batch-size 32
```

### 3. Making Predictions

Make an instant prediction for specific ambient conditions:

```bash
python -m src.predict --at 15.0 --v 40.0 --ap 1013.0 --rh 70.0
```

Batch prediction from a CSV file:

```bash
python -m src.predict --file path/to/input.csv
```

### 4. Running Unit Tests

Run automated tests via `pytest`:

```bash
pytest tests/ -v
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
