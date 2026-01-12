import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    layout="wide"
)

st.title("📈 Tesla Stock Price Prediction (LSTM)")
st.markdown("Upload Excel File | Deep Learning | Finance Domain")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_lstm_model():
    return load_model("best_model.h5")

model = load_lstm_model()

# -------------------------------
# File Upload
# -------------------------------
st.subheader("📂 Upload Tesla Stock Excel File")
uploaded_file = st.file_uploader(
    "Upload Excel file (.xlsx)",
    type=["xlsx", "csv"]
)

# -------------------------------
# Column Validation Function
# -------------------------------
def validate_columns(df):
    required_columns = ["Adj Close"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    return missing_cols

# -------------------------------
# Process Uploaded File
# -------------------------------
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Validate columns
        missing = validate_columns(df)
        if missing:
            st.error(f"❌ Missing required column(s): {', '.join(missing)}")
            st.stop()

        # Handle Date column
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)

        # Keep only Adj Close
        data = df[["Adj Close"]].dropna()

        if len(data) < 60:
            st.error("❌ At least 60 rows are required for prediction.")
            st.stop()

        st.success("✅ File uploaded and validated successfully!")
        st.dataframe(data.tail())

        # -------------------------------
        # Scaling
        # -------------------------------
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # -------------------------------
        # Create Sequences
        # -------------------------------
        def create_sequences(data, window_size=60):
            X, y = [], []
            for i in range(window_size, len(data)):
                X.append(data[i-window_size:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # -------------------------------
        # Prediction
        # -------------------------------
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)
        actual = scaler.inverse_transform(y.reshape(-1, 1))

        # -------------------------------
        # Visualization
        # -------------------------------
        st.subheader("📊 Actual vs Predicted Prices")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(actual, label="Actual Price")
        ax.plot(predictions, label="Predicted Price")
        ax.set_xlabel("Time")
        ax.set_ylabel("Stock Price")
        ax.legend()

        st.pyplot(fig)

        # -------------------------------
        # Next Day Prediction
        # -------------------------------
        st.subheader("🔮 Next Day Price Prediction")

        last_60 = scaled_data[-60:].reshape(1, 60, 1)
        next_price = model.predict(last_60)
        next_price = scaler.inverse_transform(next_price)

        st.success(f"📌 Predicted Next Day Closing Price: **${next_price[0][0]:.2f}**")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("👆 Please upload an Excel file to begin.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("""
**Model:** LSTM  
**Lookback Window:** 60 days  
**Target:** Adjusted Close Price  
**Domain:** Banking & Finance  
""")