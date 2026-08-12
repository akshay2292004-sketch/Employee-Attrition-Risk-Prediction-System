from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load("model.pkl")


# ==========================================================
# FASTAPI APP
# ==========================================================

app = FastAPI(
    title="Employee Attrition Risk Prediction API",
    description="API for predicting employee attrition risk",
    version="2.0.0"
)


# ==========================================================
# INPUT SCHEMA
# ==========================================================

class EmployeeData(BaseModel):

    last_evaluation: float
    number_project: int
    average_montly_hours: int
    time_spend_company: int
    Work_accident: int
    promotion_last_5years: int
    sales: str
    salary: str


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/")
def home():

    return {
        "message": "Employee Attrition Risk Prediction API",
        "status": "running"
    }


# ==========================================================
# PREDICTION ENDPOINT
# ==========================================================

@app.post("/predict")
def predict_attrition(employee: EmployeeData):

    # Convert API input to DataFrame
    input_data = pd.DataFrame([{
        "last_evaluation": employee.last_evaluation,
        "number_project": employee.number_project,
        "average_montly_hours": employee.average_montly_hours,
        "time_spend_company": employee.time_spend_company,
        "Work_accident": employee.Work_accident,
        "promotion_last_5years": employee.promotion_last_5years,
        "sales": employee.sales,
        "salary": employee.salary
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability of leaving
    probability = model.predict_proba(input_data)[0][1]

    # Risk category
    if probability >= 0.90:
        risk_level = "High Risk"

    elif probability >= 0.60:
        risk_level = "Medium Risk"

    elif probability >= 0.20:
        risk_level = "Low Risk"

    else:
        risk_level = "Safe"

    # Recommendation
    if probability >= 0.90:

        recommendation = (
            "Immediate retention action recommended. "
            "Review workload, job satisfaction, career growth "
            "and compensation."
        )

    elif probability >= 0.60:

        recommendation = (
            "Employee requires attention. Consider an employee "
            "check-in and review workload and career development."
        )

    elif probability >= 0.20:

        recommendation = (
            "Monitor employee engagement and provide development "
            "opportunities."
        )

    else:

        recommendation = (
            "Low immediate attrition risk. Continue regular "
            "employee engagement."
        )

    return {
        "prediction": int(prediction),
        "attrition_probability": round(float(probability), 4),
        "attrition_percentage": round(float(probability * 100), 2),
        "risk_level": risk_level,
        "recommendation": recommendation
    }