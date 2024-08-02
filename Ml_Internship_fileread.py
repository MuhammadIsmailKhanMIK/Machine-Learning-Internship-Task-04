import pandas as pd

# Specify the path to your CSV file
file_path = 'transaction_anomalies_dataset.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(df.head())