import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classificatin_report

# Loading the dataset
file_path= r'C:\Users\bibas\Downloads\GenAI\pytorch\tarabut\loan_approval_dataset.csv'

loan_data = pd.read_csv(file_path)

print(loan_data.head())

# Data cleaning

## Handle null values
print(loan_data.isnull().sum())
loan_data.dropna()

## encode label
print(loan_data.info())
loan_data.columns= loan_data.columns.str.strip()
label_encoder= LabelEncoder()
loan_data["education"]= label_encoder.fit_transform(loan_data["education"])
loan_data["self_employed"]= label_encoder.fit_transform(loan_data["self_employed"])
loan_data["loan_status"]= label_encoder.fit_transform(loan_data["loan_status"])
print(loan_data.head())

# Data preprocessing
X= loan_data.drop(["loan_id", "loan_status"], axis= 1)
y= loan_data["loan_status"]

# Check the shape of X and y
print("X shape:", X.shape)  # This should print the number of rows and columns
print("y shape:", y.shape) 
## split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state= 42)

## Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training
model= RandomForestClassifier(random_state= 42)
model.fit(X_train_scaled, y_train)

y_pred= model.predict(X_test_scaled)

# Evalute
accuracy= accuracy_score(y_test, y_pred)
report= classificatin_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification report: {report}")



