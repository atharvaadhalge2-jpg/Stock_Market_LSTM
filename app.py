# ===================== IMPORTS =====================
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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

# ===============================
# DUMMY PREDICTION (DEPLOY SAFE)
# ===============================
def dummy_predict(last_price):
    return float(last_price * 1.01)  # +1% fake growth


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

# ===================== DASHBOARD =====================
st.title("📈 Stock Price Prediction using LSTM")
st.write("Predict next-day stock price (deploy-safe demo model)")

stock = st.selectbox(
    "📊 Select Stock",
    ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
)

# ===================== LOAD DATA =====================
df = yf.download(stock, start="2018-01-01")

if df.empty:
    st.error("❌ Failed to fetch stock data.")
    st.stop()

df = df[["Close"]].dropna()

# ===================== HISTORICAL GRAPH =====================
st.subheader("📊 Historical Stock Prices")
fig, ax = plt.subplots()
ax.plot(df["Close"], label="Actual Price")
ax.legend()
st.pyplot(fig)

# ===================== PREDICTION =====================
if st.button("🚀 Predict Next Day Price"):
    last_price = float(df["Close"].values[-1])
    next_price = dummy_predict(last_price)

    st.subheader("📈 Next Day Predicted Price")
    st.success(f"₹ {next_price:.2f}")

    signal = "BUY" if next_price > last_price else "SELL"
    st.success("🟢 BUY Signal" if signal == "BUY" else "🔴 SELL Signal")

    # ---------- GRAPH ----------
    last_60 = df["Close"].values[-60:]
    pred_line = np.append(last_60[1:], next_price)

    fig2, ax2 = plt.subplots()
    ax2.plot(last_60, label="Last 60 Days")
    ax2.plot(pred_line, linestyle="dashed", label="Prediction")
    ax2.legend()
    st.pyplot(fig2)

    # ---------- SAVE ----------
    save_prediction(
        st.session_state.username,
        stock,
        round(next_price, 2),
        signal
    )

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

# ===================== ANALYTICS =====================
st.subheader("📊 Your Trading Analytics")

total, buy_count, sell_count = get_prediction_stats(st.session_state.username)

c1, c2, c3 = st.columns(3)
c1.metric("Total Predictions", total)
c2.metric("BUY Signals", buy_count)
c3.metric("SELL Signals", sell_count)

if total > 0:
    buy_pct = (buy_count / total) * 100
    st.progress(buy_pct / 100)
    st.caption(f"BUY Accuracy Trend: {buy_pct:.1f}%")

# ===================== UPDATE ACTUAL PRICE =====================
st.subheader("📅 Update Actual Price (Next Day)")

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

# ===================== ACCURACY =====================
st.subheader("🎯 Prediction Accuracy")

accuracy = get_accuracy(st.session_state.username)
st.metric("Overall Accuracy", f"{accuracy:.1f}%")
st.progress(accuracy / 100 if accuracy > 0 else 0)
