import streamlit as st
import pandas as pd
import joblib


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Employee Attrition Risk Prediction",
    page_icon="👨‍💼",
    layout="wide"
)


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()


# ==========================================================
# HEADER
# ==========================================================

st.title("Employee Attrition Risk Prediction System")

st.write(
    "Predict the probability of employee attrition using a "
    "Random Forest machine learning model."
)

st.markdown("---")


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("About the Model")

st.sidebar.write(
    """
    **Final Model:** Random Forest

    **Test Accuracy:** 84.05%

    **Test ROC-AUC:** 0.8680

    **Test PR-AUC:** 0.6967

    The model predicts whether an employee is likely to leave
    the organization.
    """
)

st.sidebar.markdown("---")

st.sidebar.subheader("Risk Levels")

st.sidebar.write(
    """
    🔴 **High Risk:** ≥ 90%

    🟠 **Medium Risk:** 60–89%

    🟡 **Low Risk:** 20–59%

    🟢 **Safe:** < 20%
    """
)


# ==========================================================
# INPUT SECTION
# ==========================================================

st.header("Employee Information")

col1, col2 = st.columns(2)


# ----------------------------------------------------------
# COLUMN 1
# ----------------------------------------------------------

with col1:

    last_evaluation = st.slider(
        "Last Evaluation",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.01,
        help="Employee's latest evaluation score."
    )

    number_project = st.number_input(
        "Number of Projects",
        min_value=1,
        max_value=20,
        value=4,
        step=1
    )

    average_montly_hours = st.number_input(
        "Average Monthly Hours",
        min_value=50,
        max_value=400,
        value=160,
        step=1
    )

    time_spend_company = st.number_input(
        "Time Spent at Company (Years)",
        min_value=1,
        max_value=20,
        value=3,
        step=1
    )


# ----------------------------------------------------------
# COLUMN 2
# ----------------------------------------------------------

with col2:

    work_accident = st.selectbox(
        "Work Accident",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    promotion_last_5years = st.selectbox(
        "Promotion in Last 5 Years",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    sales = st.selectbox(
        "Department",
        [
            "sales",
            "technical",
            "support",
            "IT",
            "product_mng",
            "marketing",
            "RandD",
            "accounting",
            "hr",
            "management"
        ]
    )

    salary = st.selectbox(
        "Salary Level",
        [
            "low",
            "medium",
            "high"
        ]
    )


# ==========================================================
# PREDICTION BUTTON
# ==========================================================

st.markdown("---")

predict_button = st.button(
    "Predict Attrition Risk",
    type="primary",
    use_container_width=True
)


# ==========================================================
# PREDICTION
# ==========================================================

if predict_button:

    # Create input DataFrame
    input_data = pd.DataFrame({
        "last_evaluation": [last_evaluation],
        "number_project": [number_project],
        "average_montly_hours": [average_montly_hours],
        "time_spend_company": [time_spend_company],
        "Work_accident": [work_accident],
        "promotion_last_5years": [promotion_last_5years],
        "sales": [sales],
        "salary": [salary]
    })

    # Model prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    percentage = probability * 100


    # ======================================================
    # RISK CLASSIFICATION
    # ======================================================

    if probability >= 0.90:

        risk_level = "High Risk"
        recommendation = (
            "Immediate retention action is recommended. "
            "Review workload, career growth, job satisfaction "
            "and compensation."
        )

    elif probability >= 0.60:

        risk_level = "Medium Risk"
        recommendation = (
            "The employee requires attention. Consider an "
            "employee check-in and review workload and career "
            "development opportunities."
        )

    elif probability >= 0.20:

        risk_level = "Low Risk"
        recommendation = (
            "Monitor employee engagement and provide career "
            "development opportunities."
        )

    else:

        risk_level = "Safe"
        recommendation = (
            "Low immediate attrition risk. Continue regular "
            "employee engagement and development."
        )


    # ======================================================
    # DISPLAY RESULTS
    # ======================================================

    st.markdown("---")

    st.header("Prediction Result")


    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Attrition Probability",
            f"{percentage:.2f}%"
        )


    with result_col2:

        st.metric(
            "Prediction",
            "Likely to Leave"
            if prediction == 1
            else "Likely to Stay"
        )


    with result_col3:

        st.metric(
            "Risk Level",
            risk_level
        )


    # ======================================================
    # PROBABILITY BAR
    # ======================================================

    st.subheader("Attrition Risk")

    st.progress(
        min(probability, 1.0)
    )

    st.write(
        f"Estimated probability of leaving: "
        f"**{percentage:.2f}%**"
    )


    # ======================================================
    # RECOMMENDATION
    # ======================================================

    st.subheader("Retention Recommendation")

    if risk_level == "High Risk":

        st.error(recommendation)

    elif risk_level == "Medium Risk":

        st.warning(recommendation)

    elif risk_level == "Low Risk":

        st.info(recommendation)

    else:

        st.success(recommendation)


    # ======================================================
    # INPUT SUMMARY
    # ======================================================

    with st.expander("View Employee Input"):

        display_data = pd.DataFrame({
            "Feature": [
                "Last Evaluation",
                "Number of Projects",
                "Average Monthly Hours",
                "Time Spent at Company",
                "Work Accident",
                "Promotion Last 5 Years",
                "Department",
                "Salary"
            ],

            "Value": [
                last_evaluation,
                number_project,
                average_montly_hours,
                time_spend_company,
                "Yes" if work_accident == 1 else "No",
                "Yes" if promotion_last_5years == 1 else "No",
                sales,
                salary
            ]
        })

        st.table(display_data)


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Employee Attrition Risk Prediction System | "
    "Random Forest Machine Learning Model"
)