# pandas-scikit-learn  
![image](https://github.com/user-attachments/assets/ae5629f4-8320-4e8a-b231-5d1649916fe3)  
# Loan Approval Prediction

This project aims to predict loan approval status using a dataset that includes various features related to applicants' profiles. A Random Forest Classifier is utilized for this classification task, and the model's performance is evaluated using accuracy and classification report metrics.


## Introduction

The goal of this project is to build a machine learning model that can effectively predict whether a loan will be approved based on several features. The dataset includes attributes such as education level, employment status, and more.

## Getting Started

To run this project, follow these steps:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/loan-approval-prediction.git
   ```
   ```
   pip install pandas numpy scikit-learn
   ```
## Data Cleaning and Preprocessing

The dataset is processed to handle missing values, encode categorical features, and scale the features before model training:

- **Missing Values**: Missing values are dropped from the dataset to ensure data integrity.
- **Categorical Variables**: Categorical variables are encoded using `LabelEncoder`, transforming them into numerical format for model compatibility.
- **Feature Standardization**: Features are standardized using `StandardScaler` to ensure that they are on the same scale, improving the model's performance.

## Model Training and Evaluation

A Random Forest Classifier is trained on the processed data. The following steps are performed during model training and evaluation:

1. The model is fit on the training data.
2. Predictions are made on the test set.
3. The model's accuracy and a detailed classification report are printed to the console, providing insights into the model's performance across different classes.


