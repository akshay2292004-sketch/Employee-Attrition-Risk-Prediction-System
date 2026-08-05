import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("Employee Attrition Risk Predictor")

# Load model and scaler once
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# Collect inputs
satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
projects = st.number_input("Number of Projects", 1, 10, 3)
hours = st.number_input("Average Monthly Hours", 50, 300, 160)
time_spent = st.number_input("Years at Company", 1, 20, 3)
accident = st.selectbox("Work Accident", [0, 1])
promotion = st.selectbox("Promotion in Last 5 Years", [0, 1])
dept = st.selectbox("Department", ["IT", "RandD", "Accounting"])
sal = st.selectbox("Salary Level", ["low", "medium", "high"])

if st.button("Predict"):
    # Encode categorical features
    payload = {
        "satisfaction_level": satisfaction,
        "last_evaluation": evaluation,
        "number_project": projects,
        "average_montly_hours": hours,
        "time_spend_company": time_spent,
        "Work_accident": accident,
        "promotion_last_5years": promotion,
        "sales_IT": 1 if dept == "IT" else 0,
        "sales_RandD": 1 if dept == "RandD" else 0,
        "sales_accounting": 1 if dept == "Accounting" else 0,
        "salary_medium": 1 if sal == "medium" else 0,
        "salary_high": 1 if sal == "high" else 0,
    }

    input_df = pd.DataFrame([payload])
    input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

    scaled_input = scaler.transform(input_df)
    prob = model.predict_proba(scaled_input)[0][1]

    if prob < 0.2:
        risk_zone = "Safe Zone"
    elif prob < 0.6:
        risk_zone = "Low-Risk Zone"
    elif prob < 0.9:
        risk_zone = "Medium-Risk Zone"
    else:
        risk_zone = "High-Risk Zone"

    st.metric("Attrition Probability", f"{prob * 100:.2f}%")
    if risk_zone == "High-Risk Zone":
        st.error(f"Risk Level: {risk_zone}")
    elif risk_zone == "Medium-Risk Zone":
        st.warning(f"Risk Level: {risk_zone}")
    else:
        st.success(f"Risk Level: {risk_zone}")
