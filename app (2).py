
import streamlit as st
import pandas as pd
import joblib

# Load saved artifacts
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn likelihood.")

# Input fields
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", [1, 0])
is_active_member = st.selectbox("Is Active Member", [1, 0])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

if st.button("Predict Churn"):
    gender_encoded = gender_encoder.transform([gender])[0]
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    input_data = pd.DataFrame([[
        credit_score, gender_encoded, age, tenure, balance,
        num_products, has_cr_card, is_active_member, estimated_salary,
        geo_germany, geo_spain
    ]], columns=[
        "CreditScore", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
        "Geography_Germany", "Geography_Spain"
    ])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"This customer is likely to churn. (Probability: {probability:.2%})")
    else:
        st.success(f"This customer is likely to stay. (Probability of churn: {probability:.2%})")
