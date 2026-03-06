# ===================== IMPORTS =====================
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from auth import (
    create_user_table,
    register_user,
    login_user,
    create_prediction_table,
    save_prediction,
    get_user_predictions,
    get_prediction_stats,
    update_actual_price,
    evaluate_predictions,
    get_accuracy
)

# ===================== INIT =====================
create_user_table()
create_prediction_table()

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ===================== LOGIN PAGE =====================
if not st.session_state.logged_in:

    st.title("🔐 Login / Signup")

    option = st.radio("Choose Option", ["Login", "Signup"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Signup":
        if st.button("Create Account"):
            if register_user(username, password):
                st.success("Account created! Please login.")
            else:
                st.error("Username already exists.")

    if option == "Login":
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password")

    st.stop()

# ===================== DISCLAIMER =====================
if "accepted_disclaimer" not in st.session_state:
    st.session_state.accepted_disclaimer = False

if not st.session_state.accepted_disclaimer:

    st.title("⚠️ Risk Disclaimer")

    st.markdown("""
    ### Stock Market Risk Disclosure

    - This application provides **AI-based stock analysis**
    - It **does NOT execute real buying or selling**
    - Predictions are based on **historical data using LSTM**
    - Stock markets are **highly volatile**
    - The developer is **not responsible for financial losses**
    - For **educational purposes only**
    """)

    if st.button("✅ I Understand and Agree"):
        st.session_state.accepted_disclaimer = True
        st.rerun()

    st.stop()

# ===================== SIDEBAR =====================
st.sidebar.success(f"Welcome {st.session_state.username}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ===================== DASHBOARD =====================
st.title("📈 Stock Price Prediction using LSTM")
st.caption("Educational AI analysis tool. Not for real trading.")

# ===================== LOAD MODEL =====================
try:
    lstm_model = load_model("model/lstm_model.h5", compile=False)
except Exception as e:
    st.error("Model failed to load: {e}")
    st.stop()

# ===================== SELECT STOCK =====================
stock = st.selectbox(
    "📊 Select Stock",
    ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
)

# ===================== LOAD DATA =====================
df = yf.download(stock, start="2018-01-01", end="2024-12-31")

if df.empty:
    st.error("Failed to fetch stock data.")
    st.stop()

df = df[["Close"]].dropna()

# ===================== SCALE DATA =====================
data = df["Close"].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# ===================== GRAPH =====================
st.subheader("📊 Historical Stock Prices")

fig, ax = plt.subplots()

ax.plot(df["Close"], label="Actual Price")

ax.legend()

st.pyplot(fig)

# ===================== PREDICTION =====================
lookback = 60

if st.button("🚀 Predict Next Day Price"):

    if len(scaled_data) < lookback:
        st.warning("Not enough data (need 60 days)")
        st.stop()

    last_60 = scaled_data[-lookback:]

    X_input = last_60.reshape(1, lookback, 1)

    predicted = lstm_model.predict(X_input)

    predicted = scaler.inverse_transform(predicted)

    next_price = predicted[0][0]

    st.subheader("📈 Next Day Predicted Price")

    st.success(f"₹ {next_price:.2f}")

    last_price = df["Close"].values[-1]

    signal = "BUY" if next_price > last_price else "SELL"

    st.subheader("📊 Analysis Result")

    if signal == "BUY":
        st.success("🟢 BUY Signal")
        st.info("Predicted price higher than current price")
    else:
        st.error("🔴 SELL Signal")
        st.warning("Predicted price lower than current price")

    save_prediction(
        st.session_state.username,
        stock,
        float(round(next_price, 2)),
        signal
    )

# ===================== MODEL PERFORMANCE =====================
X_test = []
y_test = []

for i in range(lookback, len(scaled_data)):

    X_test.append(scaled_data[i-lookback:i, 0])

    y_test.append(scaled_data[i, 0])

X_test = np.array(X_test)
y_test = np.array(y_test)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

predicted_prices = lstm_model.predict(X_test)

predicted_prices = scaler.inverse_transform(predicted_prices)

y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = math.sqrt(mean_squared_error(y_test, predicted_prices))

mae = mean_absolute_error(y_test, predicted_prices)

st.subheader("📉 Model Performance")

st.metric("RMSE", round(rmse, 2))

st.metric("MAE", round(mae, 2))

# ===================== HISTORY =====================
st.subheader("📜 Your Prediction History")

history = get_user_predictions(st.session_state.username)

if history:

    df_history = pd.DataFrame(
        history,
        columns=["Stock", "Predicted Price (₹)", "Date"]
    )

    st.dataframe(df_history)

else:
    st.info("No predictions yet")

# ===================== ANALYTICS =====================
st.subheader("📊 Your Trading Analytics")

total, buy_count, sell_count = get_prediction_stats(
    st.session_state.username
)

col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", total)

col2.metric("BUY Signals", buy_count)

col3.metric("SELL Signals", sell_count)

if total > 0:

    buy_pct = (buy_count / total) * 100

    st.progress(buy_pct / 100)

    st.caption(f"BUY Trend: {buy_pct:.1f}%")

# ===================== ACCURACY =====================
accuracy = get_accuracy(st.session_state.username)

st.subheader("🎯 Prediction Accuracy")

st.metric("Overall Accuracy", f"{accuracy:.1f}%")

# ===================== UPDATE ACTUAL PRICE =====================
st.subheader("📅 Update Actual Price")

if st.button("🔄 Fetch Today's Actual Price"):

    actual_df = yf.download(stock, period="1d")

    if not actual_df.empty:

        actual_price = float(actual_df["Close"].values[-1])

        update_actual_price(stock, actual_price)

        evaluate_predictions()

        st.success(f"Actual price updated: ₹{actual_price:.2f}")

        st.rerun()

    else:

        st.error("Failed to fetch actual price")


