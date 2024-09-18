import pandas as pd
import numpy as np

data = {
    'applicant_income': np.random.randint(2000, 10000, 100),
    'loan_amount': np.random.randint(5000, 50000, 100),
    'credit_history': np.random.choice([0, 1], size=100),
    'loan_approved': np.random.choice([0, 1], size=100)
}

loan_df = pd.DataFrame(data)

print(loan_df.head())
