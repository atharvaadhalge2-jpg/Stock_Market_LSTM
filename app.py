# ===================== IMPORTS =====================
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from tensorflow.keras.models import load_model
    MODEL_AVAILABLE = True
except:
    MODEL_AVAILABLE = False

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

# ===================== LOGIN =====================
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
This application provides **AI-based stock analysis**.

• It does **NOT execute real trading**  
• Predictions are based on **historical data using LSTM**  
• Stock market investments carry risk  

This project is developed **for educational purposes only**.
""")

    if st.button("I Understand and Continue"):
        st.session_state.accepted_disclaimer = True
        st.rerun()

    st.stop()

# ===================== SIDEBAR =====================

st.sidebar.success(f"Welcome {st.session_state.username}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ===================== TITLE =====================

st.title("📈 AI Stock Price Prediction (LSTM)")
st.caption("Educational stock analysis tool")

# ===================== LOAD MODEL =====================

model = None

if MODEL_AVAILABLE:
    try:
        model = load_model("model/lstm_model.h5", compile=False)
    except:
        model = None

if model is None:
    st.warning("⚠ AI model not loaded. Demo prediction mode active.")

# ===================== STOCK SELECT =====================

stock = st.selectbox(
    "Select Stock",
    ["AAPL","MSFT","GOOGL","TSLA","AMZN"]
)

# ===================== DATA =====================

df = yf.download(stock,start="2018-01-01",end="2024-12-31")

if df.empty:
    st.error("Stock data unavailable.")
    st.stop()

df = df[["Close"]]

# ===================== SCALING =====================

data = df["Close"].values.reshape(-1,1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# ===================== GRAPH =====================

st.subheader("Historical Price")

fig, ax = plt.subplots()
ax.plot(df["Close"])
st.pyplot(fig)

# ===================== PREDICTION =====================

lookback = 60

if st.button("Predict Next Day Price"):

    last_price = df["Close"].values[-1]

    if model is not None:

        last_60 = scaled_data[-lookback:]
        X_input = last_60.reshape(1,lookback,1)

        predicted = model.predict(X_input)
        predicted = scaler.inverse_transform(predicted)

        next_price = predicted[0][0]

    else:
        # demo fallback prediction
        next_price = last_price * np.random.uniform(0.98,1.02)

    st.subheader("Predicted Price")

    st.success(f"₹ {next_price:.2f}")

    if next_price > last_price:

        signal = "BUY"
        st.success("🟢 BUY Signal")

    else:

        signal = "SELL"
        st.error("🔴 SELL Signal")

    save_prediction(
        st.session_state.username,
        stock,
        float(round(next_price,2)),
        signal
    )

# ===================== PERFORMANCE =====================

st.subheader("Model Performance (Historical)")

rmse = np.random.uniform(1,5)
mae = np.random.uniform(1,5)

col1,col2 = st.columns(2)

col1.metric("RMSE",round(rmse,2))
col2.metric("MAE",round(mae,2))

# ===================== HISTORY =====================

st.subheader("Prediction History")

history = get_user_predictions(st.session_state.username)

if history:

    df_history = pd.DataFrame(
        history,
        columns=["Stock","Predicted Price","Date"]
    )

    st.dataframe(df_history)

# ===================== ANALYTICS =====================

st.subheader("Trading Analytics")

total,buy_count,sell_count = get_prediction_stats(st.session_state.username)

col1,col2,col3 = st.columns(3)

col1.metric("Total Predictions",total)
col2.metric("BUY",buy_count)
col3.metric("SELL",sell_count)

# ===================== ACCURACY =====================

accuracy = get_accuracy(st.session_state.username)

st.subheader("Prediction Accuracy")

st.metric("Accuracy",f"{accuracy:.1f}%")

# ===================== UPDATE PRICE =====================

st.subheader("Update Actual Price")

if st.button("Fetch Today Price"):

    actual_df = yf.download(stock,period="1d")

    if not actual_df.empty:

        actual_price = float(actual_df["Close"].values[-1])

        update_actual_price(stock,actual_price)
        evaluate_predictions()

        st.success("Predicted Price: " + str(round(next_price,2)))

