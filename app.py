import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# ------------------------------
# Load model artifacts safely
# ------------------------------
try:
    model = joblib.load("model/churn_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    encoder = joblib.load("model/encoder.pkl")
except Exception as e:
    st.error(f"❌ Model artifacts missing: {e}")
    st.stop()

# ------------------------------
# Streamlit Page Setup
# ------------------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("This app predicts whether a customer is likely to churn based on their details.")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("🧾 Enter Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

Contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

MonthlyCharges = st.sidebar.number_input("Monthly Charges", 0.0, 150.0, 70.0)
TotalCharges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 500.0)

# ------------------------------
# Convert inputs to DataFrame
# ------------------------------
input_dict = {
    'gender': gender,
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}

input_df = pd.DataFrame([input_dict])

# ------------------------------
# Encode categorical columns
# ------------------------------
for col in input_df.select_dtypes('object').columns:
    input_df[col] = encoder.fit_transform(input_df[col])

# ------------------------------
# Scale numeric values
# ------------------------------
scaled_input = scaler.transform(input_df)

# ------------------------------
# Predict Churn
# ------------------------------
st.subheader("🔍 Prediction")

if st.button("🔮 Predict Churn"):
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    with st.expander("📋 Customer Summary"):
        st.write(input_df)

    if pred == 1:
        st.error(f"⚠️ This customer is **likely to churn**. (Probability: {prob:.2%})")
    else:
        st.success(f"✅ This customer is **likely to stay**. (Probability: {prob:.2%})")

# ------------------------------
# 📈 Model Performance Dashboard
# ------------------------------
st.subheader("📊 Model Performance Over Time")

metrics_path = "model/metrics_history.csv"

if os.path.exists(metrics_path):
    metrics_df = pd.read_csv(metrics_path)
    metrics_df["timestamp"] = pd.to_datetime(metrics_df["timestamp"])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Latest Accuracy", f"{metrics_df['accuracy'].iloc[-1]:.3f}")
    with col2:
        st.metric("Latest F1-Score", f"{metrics_df['f1_score'].iloc[-1]:.3f}")

    fig = px.line(
        metrics_df,
        x="timestamp",
        y=["accuracy", "f1_score"],
        title="Model Accuracy and F1 Trends Over Time",
        markers=True,
        labels={"value": "Score", "timestamp": "Retraining Date", "variable": "Metric"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("This chart updates automatically after each retraining.")
else:
    st.warning("⚠️ No performance history found. Run a retraining cycle to generate metrics.")
