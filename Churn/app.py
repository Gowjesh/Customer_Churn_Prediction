import streamlit as st
import pandas as pd
import joblib

# Set wide page layout
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# Load model
model = joblib.load("best_model.pkl")
columns = joblib.load("columns.pkl")

st.title("🪑 Customer Churn Prediction")
st.write("Enter the customer details below to evaluate churn status.")

# Reorganize inputs into a clean 3-column layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    st.subheader("Billing Rates")
    monthly = st.number_input("Monthly Charges", 0.0, value=65.0)
    total = st.number_input("Total Charges", 0.0, value=150.0)
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

with col3:
    st.subheader("Contract Info")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    tenure = st.number_input("Tenure (Months)", 0, 72, value=12)

# Build feature dictionary
user = {}
for col in columns:
    user[col] = 0

user["SeniorCitizen"] = senior
user["tenure"] = tenure
user["MonthlyCharges"] = monthly
user["TotalCharges"] = total
user["AvgCharge"] = total / (tenure + 1)
user["SeniorSingle"] = 1 if senior == 1 and partner == "No" else 0

if gender == "Male":
    user["gender_Male"] = 1
    
if partner == "Yes":
    user["Partner_Yes"] = 1
    
if dependents == "Yes":
    user["Dependents_Yes"] = 1
    
if contract == "One year":
    user["Contract_One year"] = 1
elif contract == "Two year":
    user["Contract_Two year"] = 1
    
if paperless == "Yes":
    user["PaperlessBilling_Yes"] = 1
    
input_df = pd.DataFrame([user])

st.markdown("---")

# Predict button
if st.button("Predict Churn Status", use_container_width=True):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("Customer Will Go")
    else:
        st.success("Customer Will Stay")
