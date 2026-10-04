"""
Streamlit Web Application for PowerPredict.
Predicts net hourly electrical energy output of a Combined Cycle Power Plant (CCPP).
"""
import os
import io
import pandas as pd
import numpy as np
import streamlit as st

from src.predict import PowerPredictor
from src.dataset import FEATURE_COLUMNS

# Page configuration
st.set_page_config(
    page_title="PowerPredict | CCPP Energy Output Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        border-radius: 12px;
        padding: 24px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin: 10px 0;
    }
    .metric-val {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .metric-label {
        font-size: 0.95rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stat-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    """Load model and scaler once and cache in memory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "best_model.pt")
    scaler_path = os.path.join(base_dir, "models", "scaler.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None
    return PowerPredictor(model_path=model_path, scaler_path=scaler_path)


# Sidebar - Info & Scenario Presets
st.sidebar.title("⚡ PowerPredict Controls")

# Initialize session state for inputs if not present
if "at" not in st.session_state:
    st.session_state.at = 15.0
if "v" not in st.session_state:
    st.session_state.v = 40.0
if "ap" not in st.session_state:
    st.session_state.ap = 1013.0
if "rh" not in st.session_state:
    st.session_state.rh = 70.0


def set_preset(at, v, ap, rh):
    st.session_state.at = at
    st.session_state.v = v
    st.session_state.ap = ap
    st.session_state.rh = rh


st.sidebar.subheader("🌡️ Operating Scenarios")
st.sidebar.write("Quickly populate ambient settings:")
col_p1, col_p2 = st.sidebar.columns(2)
with col_p1:
    if st.button("❄️ Cold Morning", use_container_width=True):
        set_preset(6.5, 38.0, 1018.5, 85.0)
    if st.button("🌧️ Humid Rainy", use_container_width=True):
        set_preset(18.0, 52.0, 1011.0, 95.0)
with col_p2:
    if st.button("☀️ Hot Summer", use_container_width=True):
        set_preset(32.5, 72.0, 1007.5, 45.0)
    if st.button("⚖️ Moderate Base", use_container_width=True):
        set_preset(15.0, 40.0, 1013.0, 70.0)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Specifications")
st.sidebar.markdown("""
- **Model:** PyTorch Artificial Neural Network (MLP)
- **Architecture:** 4 → 6 → 6 → 1
- **Trainable Parameters:** 79
- **Test $R^2$ Score:** `92.97%`
- **Test RMSE:** `4.43 MW`
""")

# Main Content
st.markdown('<div class="main-header">⚡ PowerPredict: CCPP Electrical Energy Output</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict Net Hourly Electrical Energy Output (PE) of a Combined Cycle Power Plant based on ambient environmental conditions.</div>', unsafe_allow_html=True)

predictor = get_predictor()

if predictor is None:
    st.error("⚠️ Model checkpoint (`models/best_model.pt`) or Scaler (`models/scaler.joblib`) not found. Please run `python -m src.train` first.")
    st.stop()

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["⚡ Single Prediction", "📁 Batch Prediction", "📊 Model Performance & Insights"])

# ----------------- TAB 1: SINGLE PREDICTION -----------------
with tab1:
    col_input, col_result = st.columns([3, 2], gap="large")

    with col_input:
        st.subheader("Ambient Environmental Parameters")
        
        at = st.slider(
            "Ambient Temperature (AT in °C)",
            min_value=1.5,
            max_value=38.0,
            value=float(st.session_state.at),
            step=0.1,
            help="Atmospheric temperature entering the gas turbine compressor (1.81°C - 37.11°C)"
        )
        st.session_state.at = at

        v = st.slider(
            "Exhaust Vacuum (V in cm Hg)",
            min_value=25.0,
            max_value=82.0,
            value=float(st.session_state.v),
            step=0.1,
            help="Vacuum pressure at the exhaust of the steam turbine (25.36 - 81.56 cm Hg)"
        )
        st.session_state.v = v

        ap = st.slider(
            "Ambient Pressure (AP in mbar)",
            min_value=990.0,
            max_value=1035.0,
            value=float(st.session_state.ap),
            step=0.1,
            help="Atmospheric barometric pressure (992.89 - 1033.30 mbar)"
        )
        st.session_state.ap = ap

        rh = st.slider(
            "Relative Humidity (RH in %)",
            min_value=25.0,
            max_value=100.0,
            value=float(st.session_state.rh),
            step=0.1,
            help="Percentage of ambient relative humidity (25.56% - 100.16%)"
        )
        st.session_state.rh = rh

    with col_result:
        st.subheader("Predicted Output")
        
        # Make prediction
        input_sample = [[at, v, ap, rh]]
        predicted_mw = float(predictor.predict(input_sample))
        
        # Metric Card Display
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Net Hourly Electrical Output</div>
            <div class="metric-val">{predicted_mw:.2f} <span style="font-size: 1.5rem; font-weight: 500;">MW</span></div>
            <div style="font-size: 0.85rem; opacity: 0.85; margin-top: 6px;">Combined Cycle (Gas + Steam Turbines)</div>
        </div>
        """, unsafe_allow_html=True)

        # Capacity Gauge / Relative Output Progress Bar
        min_pe, max_pe = 420.0, 496.0
        norm_output = np.clip((predicted_mw - min_pe) / (max_pe - min_pe), 0.0, 1.0)
        st.write(f"**Plant Capacity Utilization:** {norm_output*100:.1f}%")
        st.progress(float(norm_output))
        st.caption(f"Historical Plant Range: {min_pe:.0f} MW (Base Min) to {max_pe:.0f} MW (Peak Max)")

        # Thermodynamic explanation
        st.markdown("#### 💡 Thermodynamic Context")
        if at > 25.0:
            st.info("🔥 **High Ambient Temperature:** Warmer air reduces air density and mass flow rate into the gas turbine, lowering output.")
        elif at < 10.0:
            st.success("❄️ **Low Ambient Temperature:** Denser cold air increases turbine mass flow, producing higher electrical output.")
        else:
            st.write("⚙️ **Moderate Temperature:** Turbine operates near nominal baseline conditions.")

# ----------------- TAB 2: BATCH PREDICTION -----------------
with tab2:
    st.subheader("Batch File Prediction")
    st.write("Upload a CSV file containing ambient measurements (`AT`, `V`, `AP`, `RH`) to generate predictions in bulk.")

    sample_template = pd.DataFrame({
        "AT": [14.96, 25.18, 5.11, 20.84],
        "V": [41.76, 62.96, 39.40, 53.77],
        "AP": [1024.07, 1020.04, 1012.16, 1017.29],
        "RH": [73.17, 59.08, 92.14, 66.84]
    })
    
    col_dl, col_up = st.columns([1, 2])
    with col_dl:
        csv_buffer = io.StringIO()
        sample_template.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=csv_buffer.getvalue(),
            file_name="ccpp_sample_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    uploaded_file = st.file_uploader("Upload CSV file for batch inference", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            missing_cols = [col for col in FEATURE_COLUMNS if col not in batch_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}. Your CSV must contain columns: {FEATURE_COLUMNS}")
            else:
                st.write(f"**Loaded {len(batch_df)} rows.** Preview:")
                st.dataframe(batch_df.head(5), use_container_width=True)

                if st.button("🚀 Run Batch Prediction", type="primary"):
                    with st.spinner("Generating predictions..."):
                        preds = predictor.predict(batch_df)
                        batch_df["Predicted_PE_MW"] = np.round(preds, 2)
                    
                    st.success(f"✅ Successfully computed {len(batch_df)} predictions!")
                    st.dataframe(batch_df.head(10), use_container_width=True)

                    # Prepare download
                    out_csv = io.StringIO()
                    batch_df.to_csv(out_csv, index=False)
                    st.download_button(
                        label="💾 Download Predictions CSV",
                        data=out_csv.getvalue(),
                        file_name="ccpp_predictions.csv",
                        mime="text/csv",
                        type="primary"
                    )
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")

# ----------------- TAB 3: MODEL PERFORMANCE & INSIGHTS -----------------
with tab3:
    st.subheader("Model Evaluation & Training Analytics")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Test R² Score", "92.97%", "High Precision")
    with col_m2:
        st.metric("Test MSE", "19.62", "-0.83 vs baseline")
    with col_m3:
        st.metric("Test RMSE", "4.43 MW", "~0.9% error")
    with col_m4:
        st.metric("Model Size", "79 Params", "Ultra-lightweight")

    st.markdown("---")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(base_dir, "assets")

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("#### 1. Convergence & Loss Minimization")
        loss_img = os.path.join(assets_dir, "Minimization of Loss.png")
        if os.path.exists(loss_img):
            st.image(loss_img, caption="Training vs Validation Loss over 100 Epochs", use_container_width=True)
            st.caption("Rapid convergence within 25 epochs; validation error stabilizes with zero overfitting.")
        else:
            st.info("Loss plot not found in assets.")

    with col_img2:
        st.markdown("#### 2. Actual vs. Predicted Output")
        fit_img = os.path.join(assets_dir, "Actual vs predicted.png")
        if os.path.exists(fit_img):
            st.image(fit_img, caption="Scatter of Predicted vs Actual Values", use_container_width=True)
            st.caption("Points cluster along the identity line (y = x) across the full plant range (420–495 MW).")
        else:
            st.info("Fit plot not found in assets.")

    st.markdown("#### 3. Error Residuals Distribution")
    res_img = os.path.join(assets_dir, "Distribution of residuals.png")
    if os.path.exists(res_img):
        st.image(res_img, caption="Histogram of Prediction Errors (Ground Truth - Prediction)", use_container_width=True)
        st.caption("Residuals follow a normal Gaussian curve centered at zero, proving unbiased predictions.")
    else:
        st.info("Residuals plot not found in assets.")

    st.markdown("---")
    st.markdown("#### 🧠 Artificial Neural Network (ANN) Architecture")
    arch_img = os.path.join(assets_dir, "nn_architecture.png")
    if os.path.exists(arch_img):
        st.image(arch_img, caption="Feedforward Dense Architecture (4 -> 6 -> 6 -> 1)", use_container_width=True)
        st.caption("Dense feedforward architecture with 79 parameters, ReLU activations, and linear regression head.")

