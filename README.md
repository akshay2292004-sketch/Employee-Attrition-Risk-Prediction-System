Employee Attrition Risk Prediction System

A machine learning-based system designed to predict employee attrition risk and help organizations identify employees who may be at risk of leaving.

Overview

Employee attrition can lead to increased recruitment costs, loss of experienced employees, and workforce instability. This project uses employee-related factors to estimate the probability of attrition and categorize employees into different risk levels.

The system provides:

Employee attrition prediction
Attrition probability
Risk-level classification
Retention recommendations
Interactive Streamlit application
FastAPI REST API
Machine Learning Models

Three classification models were evaluated:

Logistic Regression
Random Forest
Gradient Boosting
Cross-Validation Results
Model	Average ROC-AUC	Standard Deviation
Logistic Regression	0.6917	0.0120
Random Forest	0.8774	0.0021
Gradient Boosting	0.8703	0.0045

Random Forest was selected as the final model based on the cross-validation results.

Final Model Performance
Metric	Result
Test Accuracy	84.05%
Test ROC-AUC	0.8680
Test PR-AUC	0.6967
Input Features

The deployed system uses:

Last evaluation
Number of projects
Average monthly hours
Time spent at company
Work accident
Promotion in the last 5 years
Department
Salary
Risk Classification
Attrition Probability	Risk Level
Below 20%	Safe
20% – below 60%	Low Risk
60% – below 90%	Medium Risk
90% and above	High Risk

The system also provides retention recommendations based on the predicted risk level.

Deployment

The project includes two deployment components:

Streamlit
Provides an interactive user interface for entering employee information and viewing prediction results.

FastAPI
Provides a REST API for making employee attrition predictions.

Technologies
Python
Pandas
NumPy
Scikit-learn
Imbalanced-learn
Random Forest
Streamlit
FastAPI
Joblib
Git
Git LFS
Key Highlights
Compared multiple machine learning classification models
Selected Random Forest using cross-validation
Achieved 0.8774 average cross-validation ROC-AUC
Achieved 0.8680 test ROC-AUC
Implemented probability-based risk classification
Added employee retention recommendations
Developed an interactive prediction interface
Developed a REST API for deployment
Used Git LFS to manage the trained model
Project Purpose

This project demonstrates an end-to-end machine learning workflow, from employee data and model evaluation to deployment and real-world attrition risk prediction.

Disclaimer

This project is developed for educational and portfolio purposes. The predictions represent estimated attrition risk and should be used as decision-support information rather than a definitive assessment of an individual employee.
