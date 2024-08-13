import pandas as pd

# Load the CSV file
df = pd.read_csv('~/Desktop/churn.csv')

# Slice the first 10 rows
sample_df = df.head(50)

# Save the new file
sample_df.to_csv('~/Desktop/test_sample_data.csv', index=False)
