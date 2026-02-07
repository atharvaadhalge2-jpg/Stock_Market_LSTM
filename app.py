# ===================== IMPORTS =====================
from auth import update_actual_price, evaluate_predictions
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from auth import (
    create_user_table,
    register_user,
    login_user,
    create_prediction_table,
    save_prediction,
    get_user_predictions,
    get_prediction_stats
)

# ===================== INIT DB =====================
create_user_table()
create_prediction_table()

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# ===================== SESSION =====================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ===================== LOGIN / SIGNUP =====================
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
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

    st.stop()

# ===================== SIDEBAR =====================
st.sidebar.success(f"Welcome {st.session_state.username}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()

# ===================== LOAD MODEL =====================
lstm_model = load_model("model/lstm_model.keras")

# ===================== DASHBOARD =====================
st.title("📈 Stock Price Prediction using LSTM")
st.write("Predict next-day stock price using Deep Learning (LSTM)")

stock = st.selectbox(
    "📊 Select Stock",
    ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
)

# ===================== LOAD DATA =====================
df = yf.download(stock, start="2018-01-01", end="2024-12-31")

if df.empty:
    st.error("❌ Failed to fetch stock data.")
    st.stop()

df = df[["Close"]].dropna()

# ===================== SCALE DATA =====================
data = df["Close"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# ===================== HISTORICAL GRAPH =====================
st.subheader("📊 Historical Stock Prices")
fig, ax = plt.subplots()
ax.plot(df["Close"], label="Actual Price")
ax.legend()
st.pyplot(fig)

# ===================== PREDICTION =====================
lookback = 60

if st.button("🚀 Predict Next Day Price"):

    if len(scaled_data) < lookback:
        st.warning("Not enough data (need at least 60 days).")
    else:
        last_60_days = scaled_data[-lookback:]
        X_input = last_60_days.reshape(1, lookback, 1)

        predicted = lstm_model.predict(X_input)
        predicted = scaler.inverse_transform(predicted)
        next_price = predicted[0][0]

        st.subheader("📈 Next Day Predicted Price")
        st.success(f"₹ {next_price:.2f}")

        last_price = df["Close"].values[-1]
        signal = "BUY" if next_price > last_price else "SELL"
        if next_price > last_price:
            st.success("🟢 BUY Signal")
        else:
            st.error("🔴 SELL Signal")

        # -------- GRAPH ----------
        last_actual = df["Close"].values[-lookback:]
        pred_line = np.append(last_actual[1:], next_price)

        fig2, ax2 = plt.subplots()
        ax2.plot(last_actual, label="Last 60 Days")
        ax2.plot(pred_line, linestyle="dashed", label="Prediction")
        ax2.legend()
        st.pyplot(fig2)

        # -------- SAVE ----------
        signal = "BUY" if next_price > last_price else "SELL"

save_prediction(
    st.session_state.username,
    stock,
    float(round(next_price, 2)),
    signal
)


# ================== ACTUAL PRICE UPDATE ==================
actual_df = yf.download(stock, period="1d")

if not actual_df.empty:
    actual_price = float(actual_df["Close"].values[-1])

    update_actual_price(stock, actual_price)
    evaluate_predictions()

    st.success(f"✅ Actual price updated: ₹{actual_price:.2f}")






# ===================== RMSE / MAE =====================
X_test, y_test = [], []

for i in range(lookback, len(scaled_data)):
    X_test.append(scaled_data[i-lookback:i, 0])
    y_test.append(scaled_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
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
    st.info("No predictions yet.")

st.subheader("📊 Your Trading Analytics")

total, buy_count, sell_count = get_prediction_stats(st.session_state.username)

col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", total)
col2.metric("BUY Signals", buy_count)
col3.metric("SELL Signals", sell_count)

if total > 0:
    buy_pct = (buy_count / total) * 100
    st.progress(buy_pct / 100)
    st.caption(f"BUY Accuracy Trend: {buy_pct:.1f}%")


from auth import get_accuracy

accuracy = get_accuracy(st.session_state.username)

st.subheader("🎯 Prediction Accuracy")
st.metric("Overall Accuracy", f"{accuracy:.1f}%")



st.subheader("📅 Update Actual Price (Next Day)")

if st.button("🔄 Fetch Today's Actual Price"):
    actual_df = yf.download(stock, period="1d")

    if not actual_df.empty:
        actual_price = float(actual_df["Close"].values[-1])

        update_actual_price(stock, actual_price)
        evaluate_predictions()

        st.success(f"Actual price updated: ₹{actual_price}")
        st.rerun()
    else:
        st.error("Failed to fetch actual price")

st.subheader("🎯 Prediction Accuracy")

correct = sum(1 for row in history if row[-2] == 1)
total = len(history)

if total > 0:
    accuracy = (correct / total) * 100
    st.metric("Overall Accuracy", f"{accuracy:.1f}%")
    st.progress(accuracy / 100)
else:
    st.info("No evaluated predictions yet.")

