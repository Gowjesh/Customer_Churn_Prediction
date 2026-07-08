# Data Manipulation
import pandas as pd
import numpy as np

import joblib

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Load Dataset

df = pd.read_csv("C:\\Users\\Gowje\\OneDrive\\Attachments\\Churn\\WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("="*60)
print("First Five Rows")
print(df.head())


print("="*60)
print("Dataset Information")
print(df.info())

print("="*60)
print("Statistical Summary")
print(df.describe(include="all"))

# Missing Values

print("="*60)
print("Missing Values")
print(df.isnull().sum())

# Duplicate Values

print("="*60)
print("Duplicate Rows")
print(df.duplicated().sum())

# Convert TotalCharges into Numeric

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")

# Fill missing values

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Remove duplicates

df.drop_duplicates(inplace=True)

print("Dataset Shape After Cleaning")
print(df.shape)

# EDA

# Churn Distribution

plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.show()

# Gender vs Churn

plt.figure(figsize=(6,4))
sns.countplot(x="gender", hue="Churn", data=df)
plt.title("Gender vs Churn")
plt.show()

# Total Charges Distribution

plt.figure(figsize=(7,4))
sns.histplot(df["TotalCharges"], bins=30, kde=True)
plt.title("Total Charges Distribution")
plt.show()

# Monthly Charges vs Churn

plt.figure(figsize=(7,4))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.show()

# Correlation Heatmap

numeric_df = df.select_dtypes(include=["int64","float64"])

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# DATA PREPROCESSING

df.drop("customerID", axis=1, inplace=True)

# Encode Target Variable
df["Churn"] = df["Churn"].map({"Yes":1,"No":0})

# One-Hot Encoding
df = pd.get_dummies(df, drop_first=True, dtype=int)

print(df.head())

# FEATURE ENGINEERING

df["AvgCharge"] = df["TotalCharges"] / (df["tenure"] + 1)

df["SeniorSingle"] = (
    (df["SeniorCitizen"] == 1) &
    (df["Partner_Yes"] == 0)
).astype(int)

df = df.fillna(0)

# SPLIT FEATURES AND TARGET

X = df.drop("Churn", axis=1)
y = df["Churn"]

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# FEATURE SCALING

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LOGISTIC REGRESSION

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
pred_lr = lr.predict(X_test_scaled)

# KNN

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
pred_knn = knn.predict(X_test_scaled)

# DECISION TREE

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)

# RANDOM FOREST

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print("All Models Trained Successfully")

# MODEL EVALUATION FUNCTION

def evaluate_model(model_name, y_true, y_pred):

    print("\n" + "="*60)
    print(model_name)
    print("="*60)

    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))

    print("\nClassification Report\n")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues")

    plt.title(model_name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# EVALUATE ALL MODELS

evaluate_model("Logistic Regression", y_test, pred_lr)
evaluate_model("KNN", y_test, pred_knn)
evaluate_model("Decision Tree", y_test, pred_dt)
evaluate_model("Random Forest", y_test, pred_rf)

# ROC AUC SCORE

models = {
    "Logistic Regression": (lr, X_test_scaled),
    "KNN": (knn, X_test_scaled),
    "Decision Tree": (dt, X_test),
    "Random Forest": (rf, X_test)
}

print("\nROC AUC Scores")

for name, (model, data) in models.items():
    probability = model.predict_proba(data)[:,1]
    score = roc_auc_score(y_test, probability)
    print(name, ":", round(score,4))

# GRID SEARCH CV

param_grid = {
    "n_estimators":[100,200,300],
    "max_depth":[5,10,15,None],
    "min_samples_split":[2,5,10],
    "min_samples_leaf":[1,2,4]
}

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest Parameters")
print(grid.best_params_)

print("\nBest Accuracy")
print(grid.best_score_)

best_model = grid.best_estimator_
best_prediction = best_model.predict(X_test)

model_scores = {
    "Logistic Regression": accuracy_score(y_test, pred_lr),
    "KNN": accuracy_score(y_test, pred_knn),
    "Decision Tree": accuracy_score(y_test, pred_dt),
    "Random Forest": accuracy_score(y_test, pred_rf),
    "Tuned Random Forest": accuracy_score(y_test, best_prediction)
}

print("\nModel Scores")

best_model_name = max(model_scores, key=model_scores.get)
best_accuracy = model_scores[best_model_name]

print("\n" + "="*60)
print("BEST MODEL")
print("="*60)
print("Model Name :", best_model_name)
print("Accuracy   :", round(best_accuracy, 4))

if best_model_name == "Tuned Random Forest":
    print("Best Parameters :", grid.best_params_)

print("="*60)

# Evaluate Best Model

evaluate_model(
    "Tuned Random Forest",
    y_test,
    best_prediction
)

# FEATURE IMPORTANCE

importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features")

print(importance.head(10))

plt.figure(figsize=(10,6))

importance.head(10).plot(kind="bar")

plt.title("Top 10 Important Features")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.show()

# BUSINESS RECOMMENDATIONS

print("\n" + "="*60)
print("BUSINESS RECOMMENDATIONS")
print("="*60)

print("""
1. Customers with Month-to-Month contracts have a higher chance of churn.
2. Customers with short tenure should receive welcome offers and loyalty benefits.
3. Customers with higher Monthly Charges are more likely to leave.
4. Improve customer service for Fiber Optic users.
5. Encourage customers to switch from Month-to-Month to One-Year or Two-Year contracts.
6. Reward long-term customers with discounts and loyalty programs.
7. Use the trained Random Forest model to identify customers who are likely to churn and target them with personalized retention campaigns.
""")

joblib.dump(best_model, "best_model.pkl")
if best_model_name not in ["Tuned Random Forest","Random Forest","Decision Tree"]:
    joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("Model Saved Successfully")