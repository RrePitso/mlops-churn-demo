import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessing artifacts
model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/encoder.pkl")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("This app predicts whether a customer is likely to churn based on their details.")

# Define input fields
st.sidebar.header("Enter Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
Contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
MonthlyCharges = st.sidebar.number_input("Monthly Charges", 0.0, 150.0, 70.0)
TotalCharges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 500.0)

# Convert categorical variables using encoder
input_dict = {
    'gender': gender,
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'PhoneService': PhoneService,
    'InternetService': InternetService,
    'Contract': Contract,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}

input_df = pd.DataFrame([input_dict])

# Encode categorical columns like training
for col in input_df.select_dtypes('object').columns:
    input_df[col] = encoder.fit_transform(input_df[col])

# Scale numeric values
scaled_input = scaler.transform(input_df)

# Predict
if st.button("🔮 Predict Churn"):
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    if pred == 1:
        st.error(f"⚠️ This customer is likely to churn. (Probability: {prob:.2%})")
    else:
        st.success(f"✅ This customer is likely to stay. (Probability: {prob:.2%})")
