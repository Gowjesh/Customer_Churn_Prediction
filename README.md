# Customer Churn Prediction

## Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a telecom customer is likely to leave the service or continue as a customer. The project analyzes customer information such as tenure, contract type, monthly charges, internet service, and payment method to classify customers as churn or non-churn.

The objective is to help telecom companies identify customers who are at risk of leaving so that they can take appropriate retention strategies.

---

## Features

- Data preprocessing and cleaning
- Missing value handling
- Feature engineering
- One-Hot Encoding for categorical variables
- Feature scaling using StandardScaler
- Training multiple Machine Learning models
- Hyperparameter tuning using GridSearchCV
- Model evaluation using different performance metrics
- Streamlit web application for prediction
- Model saving and loading using Joblib

---

## Dataset

Dataset Used:

WA_Fn-UseC_-Telco-Customer-Churn.csv

Target Variable:

- Yes → Customer Churn
- No → Customer Retained

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit

---

## Data Preprocessing

The following preprocessing steps were performed:

- Removed duplicate records
- Converted TotalCharges into numeric format
- Filled missing values
- Dropped CustomerID column
- Encoded categorical features
- Standardized numerical features
- Split the dataset into training and testing sets

---

## Feature Engineering

Additional features were created to improve prediction performance, including:

- Average Charge
- Senior Citizen without Partner indicator

---

## Machine Learning Models

The following models were trained and compared:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

Random Forest achieved the best performance after hyperparameter tuning.

---

## Hyperparameter Tuning

GridSearchCV was used to optimize the Random Forest model and obtain the best hyperparameters.

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

---

## Streamlit Application

The project includes a Streamlit application that allows users to:

- Enter customer details
- Predict customer churn
- Display prediction results instantly

Run the application using:

```bash
streamlit run app.py
```

---

## Project Structure

```
Customer-Churn-Prediction/
│
├── dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── columns.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
```

Move to the project directory:

```bash
cd customer-churn-prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Workflow

1. Load the telecom customer dataset.
2. Perform data preprocessing and cleaning.
3. Apply feature engineering.
4. Train and evaluate multiple machine learning models.
5. Save the best-performing model.
6. Load the trained model in the Streamlit application.
7. Enter customer information.
8. Predict whether the customer will churn.

---

## Future Enhancements

- Improve prediction accuracy with advanced algorithms.
- Deploy the application on cloud platforms.
- Add Explainable AI (XAI) techniques.
- Support real-time predictions using APIs.
- Create an interactive dashboard with more visualizations. 
