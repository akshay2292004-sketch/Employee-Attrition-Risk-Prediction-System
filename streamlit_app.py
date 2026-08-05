import streamlit as st
import requests

st.title("Employee Attrition Risk Predictor")

# Collect inputs
satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
projects = st.number_input("Number of Projects", 1, 10, 3)
hours = st.number_input("Average Monthly Hours", 50, 300, 160)
time_spent = st.number_input("Years at Company", 1, 20, 3)
accident = st.selectbox("Work Accident", [0, 1])
promotion = st.selectbox("Promotion in Last 5 Years", [0, 1])

dept = st.selectbox(
    "Department",
    ["IT", "RandD", "Accounting"]
)

sal = st.selectbox(
    "Salary Level",
    ["low", "medium", "high"]
)


if st.button("Predict"):

    payload = {
        "satisfaction_level": satisfaction,
        "last_evaluation": evaluation,
        "number_project": projects,
        "average_montly_hours": hours,
        "time_spend_company": time_spent,
        "Work_accident": accident,
        "promotion_last_5years": promotion,

        "sales_IT": 0,
        "sales_RandD": 0,
        "sales_accounting": 0,

        "salary_medium": 0,
        "salary_high": 0
    }


    # Department encoding
    if dept == "IT":
        payload["sales_IT"] = 1

    elif dept == "RandD":
        payload["sales_RandD"] = 1

    elif dept == "Accounting":
        payload["sales_accounting"] = 1


    # Salary encoding
    if sal == "medium":
        payload["salary_medium"] = 1

    elif sal == "high":
        payload["salary_high"] = 1


    st.subheader("Employee Details")
    st.json(payload)


    try:

        with st.spinner("Predicting employee attrition risk..."):

            response = requests.post(
                "http://127.0.0.1:8080/predict",
                json=payload
            )


        if response.status_code == 200:

            result = response.json()

            probability = result["attrition_probability"]

            st.subheader("Prediction Result")

            st.metric(
                "Attrition Probability",
                f"{probability * 100:.2f}%"
            )


            if result["risk_zone"] == "High-Risk Zone":
                st.error(
                    f"Risk Level: {result['risk_zone']}"
                )

            elif result["risk_zone"] == "Medium-Risk Zone":
                st.warning(
                    f"Risk Level: {result['risk_zone']}"
                )

            else:
                st.success(
                    f"Risk Level: {result['risk_zone']}"
                )


        else:
            st.error(
                f"API Error {response.status_code}"
            )
            st.write(response.text)


    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to FastAPI server. Start backend using uvicorn."
        )

    except Exception as e:

        st.error(
            f"Unexpected Error: {e}"
        )