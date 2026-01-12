![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

# 📈 Tesla Stock Price Prediction using SimpleRNN & LSTM

This project demonstrates **time-series forecasting of Tesla (TSLA) stock prices** using **Deep Learning models (SimpleRNN and LSTM)** and deploys the trained model using a **Streamlit web application**.

The solution is designed for **banking & finance analytics**, focusing on historical price modeling and comparative performance evaluation.

---

## 🚀 Project Overview

- **Objective:** Predict Tesla stock prices using historical data
- **Models Used:**  
  - SimpleRNN  
  - LSTM (Long Short-Term Memory)
- **Deployment:** Streamlit Web App
- **Domain:** Banking & Finance

---

## 📂 Dataset

- **Source:** Tesla historical stock price data (`TSLA.csv`)
- **Records:** 2416 daily entries
- **Columns:**
  - Date
  - Open
  - High
  - Low
  - Close
  - Adj Close *(Target Variable)*
  - Volume

> **Note:**  
> `Adj Close` is used as the target variable as it reflects the true closing price adjusted for corporate actions.

---

## 🧠 Approach & Workflow

### 1️⃣ Data Preprocessing
- Convert `Date` to datetime and set as index
- Select `Adj Close` as the target variable
- Normalize data using **MinMaxScaler**
- Create time-series sequences (60-day lookback window)

### 2️⃣ Model Development
- Built two deep learning models using Keras:
  - **SimpleRNN**
  - **LSTM**
- Architecture:
  - RNN/LSTM layer
  - Dropout (to prevent overfitting)
  - Dense output layer
- Loss Function: **Mean Squared Error (MSE)**
- Optimizer: **Adam**

### 3️⃣ Model Training
- Train-test split (80/20)
- EarlyStopping to avoid overfitting
- ModelCheckpoint to save the best model

### 4️⃣ Model Evaluation
- Compare predicted vs actual prices
- Evaluate performance using **MSE**
- Visualize predictions using Matplotlib

### 5️⃣ Deployment
- Deploy trained **LSTM model** using **Streamlit**
- Users can upload:
  - `.csv` or `.xlsx` files
- Input validation:
  - Required column: `Adj Close`
  - Minimum 60 rows required

---

## 📊 Model Comparison

| Model | Performance |
|-----|------------|
| SimpleRNN | Faster but struggles with long-term dependencies |
| LSTM | Better accuracy and trend capture |

**Result:**  
👉 LSTM consistently outperforms SimpleRNN in stock price prediction.

---

## 🖥 Streamlit Application Features

- Upload **CSV or Excel** files
- Automatic column validation
- Actual vs Predicted price visualization
- Next-day stock price prediction
- Error handling for invalid files

---

## 📁 Project Structure

├── TSLA.csv

├── TSLA_test_upload.xlsx

├── lstm_model.h5

├── streamlit_app.py

├── requirements.txt

├── README.md


---

## ▶️ How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Streamlit App

```bash
streamlit run app.py
```

### 3. Upload Dataset

* Use TSLA_test_upload.xlsx
* Ensure Adj Close column exists

## 📦 Requirements

* Python 3.8+
* TensorFlow
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* OpenPyXL

## ⚠️ Limitations

* Stock prices are influenced by external factors (news, macroeconomics)
* Model relies only on historical price data
* Not suitable for real-time trading decisions

## 🔮 Future Enhancements

* Add technical indicators (RSI, MACD)
* Include volume and sentiment analysis
* Use GRU / Bi-LSTM / Transformers

## 📌 Conclusion

This project showcases how deep learning models can be effectively applied to financial time-series forecasting. While LSTM outperforms SimpleRNN, incorporating additional financial indicators can further improve prediction robustness.