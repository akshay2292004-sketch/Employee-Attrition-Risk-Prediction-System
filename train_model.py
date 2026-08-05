import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('HR_comma_sep.csv')


# Step 1: Data Quality Checks
print("Missing values:\n", data.isnull().sum())

# Step 2: EDA
# Convert categorical variables ('sales' and 'salary') to numerical format
data = pd.get_dummies(data, columns=['sales', 'salary'], drop_first=True)

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Distribution plots
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.histplot(data['satisfaction_level'], kde=True)
plt.title('Employee Satisfaction')

plt.subplot(1, 3, 2)
sns.histplot(data['last_evaluation'], kde=True)
plt.title('Employee Evaluation')

plt.subplot(1, 3, 3)
sns.histplot(data['average_montly_hours'], kde=True)
plt.title('Average Monthly Hours')
plt.show()

# Project count bar plot by 'left'
plt.figure(figsize=(8, 6))
sns.countplot(x='number_project', hue='left', data=data)
plt.title('Project Count of Employees who Left vs Stayed')
plt.show()

# Step 3: Clustering of Employees Who Left
left_data = data[data['left'] == 1][['satisfaction_level', 'last_evaluation']]
kmeans = KMeans(n_clusters=3, random_state=42)
left_data['Cluster'] = kmeans.fit_predict(left_data)

plt.figure(figsize=(8, 6))
sns.scatterplot(x='satisfaction_level', y='last_evaluation', hue='Cluster', data=left_data, palette='viridis')
plt.title('Employee Clusters based on Satisfaction and Evaluation')
plt.show()

# Step 4: Handle Class Imbalance with SMOTE
X = data.drop('left', axis=1)
y = data['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 5: Scale the data
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Step 6: Model Training & Evaluation
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),  # Increased max_iter for convergence
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

for model_name, model in models.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='roc_auc')
    print(f"{model_name} AUC Scores: {cv_scores}")
    print(f"{model_name} Average AUC: {np.mean(cv_scores)}")


# Train the model
best_model = RandomForestClassifier(random_state=42)
best_model.fit(X_train_resampled, y_train_resampled)

# Predict
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Other evaluation metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_prob))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="AUC = {:.3f}".format(roc_auc_score(y_test, y_prob)))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 7: Retention Strategy - Probability Zones
y_prob_test = best_model.predict_proba(X_test)[:, 1]
risk_zones = pd.cut(y_prob_test, bins=[0, 0.2, 0.6, 0.9, 1.0], labels=["Safe Zone", "Low-Risk Zone", "Medium-Risk Zone", "High-Risk Zone"])
X_test = pd.DataFrame(X_test, columns=X.columns)  # Convert back to DataFrame to add 'Risk Zone'
X_test['Risk Zone'] = risk_zones

print(X_test[['Risk Zone']].value_counts())

import joblib
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

