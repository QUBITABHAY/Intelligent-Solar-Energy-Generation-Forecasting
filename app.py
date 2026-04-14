import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import mean_absolute_error, r2_score
from src.feature_aligner import align_features
from src.tools import analyze_forecast, identify_risks
from src.agent import agent
from src.report_gen import generate_report
from src.rag import get_retriever

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

@st.cache_resource
def load_retriever():
    return get_retriever()

model, scaler = load_artifacts()
retriever = load_retriever()

st.title("Solar Energy Generation Forecasting")
st.write("Upload the dataset and visualize solar power generation trends.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = None
if "forecast_summary" not in st.session_state:
    st.session_state.forecast_summary = None

if model is None:
    st.error("Model artifacts not found. Please run the training script first.")
    st.stop()

# --- Quick Forecast Sidebar ---
st.sidebar.header("⚡ Quick Forecast")
with st.sidebar.form("quick_forecast_form"):
    temp = st.slider("Temperature (°C)", min_value=-10.0, max_value=45.0, value=15.0)
    humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=51.0)
    cloud_cover = st.slider("Cloud Cover (%)", min_value=0.0, max_value=100.0, value=34.0)
    irradiance = st.slider("Solar Irradiance (W/m²)", min_value=0.0, max_value=1200.0, value=388.0)
    wind_speed = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=16.0)
    zenith = st.slider("Zenith Angle (°)", min_value=0.0, max_value=90.0, value=60.0)
    
    predict_button = st.form_submit_button("Predict")

if predict_button:
    input_data = pd.DataFrame([{
        'temp': temp,
        'humidity': humidity,
        'cloud_cover': cloud_cover,
        'irradiance': irradiance,
        'wind_speed': wind_speed,
        'zenith': zenith
    }])
    
    try:
        aligned_features = align_features(input_data)
        scaled_features = scaler.transform(aligned_features)
        prediction = model.predict(scaled_features)[0]
        st.sidebar.success(f"Predicted Power: **{prediction:.2f} kW**")
    except Exception as e:
        st.sidebar.error("An error occurred during prediction.")
        with st.sidebar.expander("Error Details"):
            st.exception(e)

# --- Grid Assistant Sidebar ---
st.sidebar.markdown("---")
st.sidebar.header("🤖 Grid Assistant")
st.sidebar.write("Search knowledge base for battery, grid, and solar insights.")

user_query = st.sidebar.text_input("Ask a question:", placeholder="e.g., How to optimize battery storage?")

if user_query:
    with st.sidebar:
        with st.spinner("Searching knowledge base..."):
            results = retriever.retrieve(user_query, k=3)
            
            if not results:
                st.warning("No relevant information found.")
            else:
                for res in results:
                    with st.expander(f"📄 {res['source']} (Match: {res['similarity']:.2%})"):
                        st.markdown(res['text'])
# Create main tabs
tab_forecast, tab_assistant = st.tabs(["📈 Forecasting", "🤖 AI Assistant"])

with tab_forecast:
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
        
        # Check if target exists (case-insensitive to handle variations)
        actual_col = None
        for col in df.columns:
            if col.strip().lower().replace(' ', '_') == TARGET_COL:
                actual_col = col
                break
        has_actual = actual_col is not None
        
        # Use Feature Aligner to handle any CSV format (different names, missing columns)
        try:
            aligned_features = align_features(df)
            scaled_features = scaler.transform(aligned_features)
            predictions = model.predict(scaled_features)
            df['Predicted Power (kW)'] = predictions
        except Exception as e:
            st.error(f"Error during prediction: {e}.")
            st.stop()

        # Store data in session state for AI Assistant
        st.session_state.forecast_df = df.copy()
        try:
            st.session_state.forecast_summary = analyze_forecast(df)
        except Exception as e:
            st.warning(f"Unable to analyze forecast: {e}")
            st.session_state.forecast_summary = None

        if has_actual:
            mae = mean_absolute_error(df[actual_col], predictions)
            r2 = r2_score(df[actual_col], predictions)
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
                ax.plot(df.index[:100], df[actual_col].iloc[:100], label='Actual', color='blue', alpha=0.7)
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

with tab_assistant:
    st.subheader("🤖 Local Intelligence — Grid Optimization")
    st.info("💡 **Local Mode**: This assistant uses your local `all-MiniLM-L6-v2` model for private data analysis.")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask about solar generation, grid balancing, or energy optimization...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate response using agent
        if st.session_state.forecast_df is not None:
            with st.spinner("Analyzing forecast..."):
                try:
                    state = {
                        "forecast_df": st.session_state.forecast_df,
                        "user_query": user_input,
                        "forecast_summary": {},
                        "risks": [],
                        "guidelines": "",
                        "recommendation": ""
                    }
                    
                    result = agent.invoke(state)
                    assistant_response = result.get("recommendation", "Unable to generate recommendation.")
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                    with st.chat_message("assistant"):
                        st.write(assistant_response)
                    
                    # Store summary for report generation
                    st.session_state.forecast_summary = result.get("forecast_summary", st.session_state.forecast_summary)
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    with st.chat_message("assistant"):
                        st.error(error_msg)
        else:
            st.warning("⚠️ Please upload and forecast data in the Forecasting tab first.")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Generate Grid Report"):
            if st.session_state.forecast_df is not None:
                with st.spinner("Generating comprehensive report..."):
                    try:
                        state = {
                            "forecast_df": st.session_state.forecast_df,
                            "user_query": "Generate a comprehensive grid optimization report",
                            "forecast_summary": {},
                            "risks": [],
                            "guidelines": "",
                            "recommendation": ""
                        }
                        
                        result = agent.invoke(state)
                        report_text = result.get("recommendation", "")
                        
                        st.success("✅ Report generated!")
                        st.markdown(report_text)
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
            else:
                st.error("❌ No forecast data available. Upload data in the Forecasting tab first.")
    
    with col2:
        if st.session_state.forecast_df is not None:
            try:
                # Prepare data for the one-click download
                forecast_df = st.session_state.forecast_df
                
                # We generate the report data for the download button. 
                # Since we are in local mode, this is fast enough for direct use.
                with st.spinner("Preparing download..."):
                    forecast_summary = analyze_forecast(forecast_df)
                    risks = identify_risks(forecast_df)
                    
                    state = {
                        "forecast_df": forecast_df,
                        "user_query": "Generate a comprehensive grid optimization report",
                        "forecast_summary": forecast_summary,
                        "risks": risks,
                        "guidelines": "",
                        "recommendation": ""
                    }
                    result = agent.invoke(state)
                    recommendation = result.get("recommendation", "")
                    
                    pdf_bytes = generate_report(
                        recommendation=recommendation,
                        forecast_summary=forecast_summary,
                        risks=risks
                    )
                
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name="solar_grid_optimization_report.pdf",
                    mime="application/pdf",
                    key="direct_pdf_download"
                )
            except Exception as e:
                st.error(f"Error preparing PDF: {str(e)}")
        else:
            st.button("📥 Download PDF Report", disabled=True, help="Upload data first to enable download.")
