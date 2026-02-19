import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Solar Energy Forecasting", layout="wide")

# constants
MODEL_PATH = 'models/random_forest_model.joblib'
SCALER_PATH = 'models/scaler.joblib'
MAX_FILE_SIZE_MB = 10
TARGET_COL = 'generated_power_kw'

# load model and scaler
@st.cache_resource
def load_artifacts():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None

model, scaler = load_artifacts()

st.title("Solar Energy Generation Forecasting")
st.write("Upload the dataset and visualize solar power generation trends.")

if model is None:
    st.error("Model artifacts not found. Please run the training script first.")
    st.stop()

uploaded_file = st.file_uploader("Upload your solar dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    # File size validation
    file_size_mb = uploaded_file.size / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large! Please upload a CSV smaller than {MAX_FILE_SIZE_MB} MB.")
        st.stop()

    # Safe CSV reading
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    
    st.subheader("Results")
    
    # Check if target exists 
    has_actual = TARGET_COL in df.columns
    
    
    features = df.drop(TARGET_COL, axis=1) if has_actual else df.copy()
    
    # Ensure columns match scaler expectations (simple check)
    try:
        scaled_features = scaler.transform(features)
        predictions = model.predict(scaled_features)
        df['Predicted Power (kW)'] = predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}. Please ensure the columns match the training data.")
        st.stop()


    if has_actual:
        mae = mean_absolute_error(df[TARGET_COL], predictions)
        r2 = r2_score(df[TARGET_COL], predictions)
        col1, col2 = st.columns(2)
        col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        col2.metric("R² Score", f"{r2:.4f}")

    # dataset preview with predictions
    st.subheader("Dataset Preview (with Predictions)")
    st.dataframe(df.head(20))

    # visualizations
    st.subheader("Result Visualization")
    
    tab1, tab2 = st.tabs(["Actual vs Predicted", "Forecast View"])
    
    with tab1:
        if has_actual:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df.index[:100], df[TARGET_COL].iloc[:100], label='Actual', color='blue', alpha=0.7)
            ax.plot(df.index[:100], df['Predicted Power (kW)'].iloc[:100], label='Predicted', color='orange', linestyle='--', alpha=0.9)
            ax.set_title("Actual vs Predicted Solar Power (First 100 samples)")
            ax.set_xlabel("Time Index")
            ax.set_ylabel("Power (kW)")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Actual data column not found. Comparison plot unavailable.")

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['Predicted Power (kW)'], label='Forecast', color='orange')
        ax.set_title("Solar Power Generation Forecast Curve")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Predicted Power (kW)")
        ax.legend()
        st.pyplot(fig)

    # dataset Info
    with st.expander("View Dataset Info"):
        st.write("Shape:", df.shape)
        st.write("Columns:", list(df.columns))
        st.write(df.describe())

else:
    st.info("Upload a CSV file to begin.")
