import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Solar Energy Forecasting", layout="wide")

st.title("Solar Energy Generation Forecasting")
st.write("Upload the dataset and visualize solar power generation trends.")

uploaded_file = st.file_uploader("Upload your solar dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))

    st.subheader("Dataset Info")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))

    # Try plotting first numeric column (just for now)
    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) > 0:
        col = st.selectbox("Select a numeric column to plot", numeric_cols)

        st.subheader(f"Plot of {col}")
        fig, ax = plt.subplots()
        ax.plot(df[col].values)
        ax.set_xlabel("Index")
        ax.set_ylabel(col)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found to plot.")
else:
    st.info("Upload a CSV file to begin.")
