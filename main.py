import pandas as pd
import numpy as np

file_path= r'C:\Users\bibas\Downloads\GenAI\pytorch\tarabut\loan_approval_dataset.csv'

loan_data = pd.read_csv(file_path)

print(loan_data.head())
