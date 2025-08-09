import streamlit as st
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler

# Title
st.set_page_config(page_title="Crypto Volatility Predictor")
st.title("📉 Cryptocurrency Volatility Predictor")

st.markdown("Enter crypto market details below to predict **volatility** using your trained model.")

# Input fields
open_price = st.number_input("Open Price", min_value=0.0)
high = st.number_input("High Price", min_value=0.0)
low = st.number_input("Low Price", min_value=0.0)
close = st.number_input("Close Price", min_value=0.0)
volume = st.number_input("Volume", min_value=0.0)
market_cap = st.number_input("Market Cap", min_value=1.0)

# Predict button
if st.button("Predict Volatility"):
    try:
        # Derived features
        liquidity_ratio = volume / market_cap
        rolling_volatility = (high - low) / open_price

        # Arrange input
        features = np.array([[open_price, high, low, close, volume, market_cap, liquidity_ratio, rolling_volatility]])

        # Scale input (for demo: fit a scaler on-the-fly)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Load model
        model = load("model.joblib")

        # Predict
        prediction = model.predict(features_scaled)

        st.success(f"📈 Predicted Volatility: `{prediction[0]:.4f}`")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
