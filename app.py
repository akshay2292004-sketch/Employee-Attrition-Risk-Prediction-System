from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

# Input schema
class EmployeeData(BaseModel):
    satisfaction_level: float
    last_evaluation: float
    number_project: int
    average_montly_hours: int
    time_spend_company: int
    Work_accident: int
    promotion_last_5years: int
    # Dummy variables for sales
    sales_IT: int = 0
    sales_RandD: int = 0
    sales_accounting: int = 0
    # Dummy variables for salary
    salary_medium: int = 0
    salary_high: int = 0

@app.post("/predict")
def predict(data: EmployeeData):
    # Convert input to dict and DataFrame
    input_dict = data.dict()
    print("\n--- Incoming Payload ---")
    print(input_dict)

    input_df = pd.DataFrame([input_dict])

    # Align columns with scaler’s expected order
    expected_cols = scaler.feature_names_in_
    print("\n--- Scaler Expects Columns ---")
    print(expected_cols)

    input_df = input_df.reindex(columns=expected_cols, fill_value=0)
    print("\n--- Aligned DataFrame ---")
    print(input_df)

    # Scale numeric + dummy features
    scaled_input = scaler.transform(input_df)

    # Predict probability
    prob = model.predict_proba(scaled_input)[0][1]
    print("\n--- Prediction Probability ---")
    print(prob)

    # Risk zone classification
    if prob < 0.2:
        risk_zone = "Safe Zone"
    elif prob < 0.6:
        risk_zone = "Low-Risk Zone"
    elif prob < 0.9:
        risk_zone = "Medium-Risk Zone"
    else:
        risk_zone = "High-Risk Zone"

    print("\n--- Risk Zone ---")
    print(risk_zone)

    return {
        "attrition_probability": float(prob),
        "risk_zone": risk_zone
    }
