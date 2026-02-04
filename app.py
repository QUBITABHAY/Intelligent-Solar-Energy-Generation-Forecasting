import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Solar Energy Forecasting", layout="wide")

st.title("Solar Energy Generation Forecasting")
st.write("Upload the dataset and visualize solar power generation trends.")

MAX_FILE_SIZE_MB = 10

uploaded_file = st.file_uploader("Upload your solar dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    # File size validation
    file_size_mb = uploaded_file.size / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large! Please upload a CSV smaller than {MAX_FILE_SIZE_MB} MB.")
        st.stop()

    # Safe CSV reading (try-except)

    try:
        df = pd.read_csv(uploaded_file)

    except pd.errors.EmptyDataError:
        st.error("Uploaded file is empty. Please upload a valid CSV file.")
        st.stop()

    except pd.errors.ParserError:
        st.error("CSV parsing error. Your file may be malformed or not a proper CSV.")
        st.stop()

    except UnicodeDecodeError:
        st.error("Encoding error. Please save your CSV in UTF-8 format and upload again.")
        st.stop()

    except Exception as e:
        st.error(f"Something went wrong while reading the CSV: {e}")
        st.stop()


    # Dataset Preview

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))


    # Dataset Info

    st.subheader("Dataset Info")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))


    # Plotting

    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) > 0:
        st.subheader("Visualization")

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
