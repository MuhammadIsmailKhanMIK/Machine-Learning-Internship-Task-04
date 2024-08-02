import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 1. Import data and check null values, check column info and the descriptive statistics of the data.
data = pd.read_csv('transaction_anomalies_dataset.csv')
print(data.info())
print(data.describe())

# Handle missing values
numeric_columns = data.select_dtypes(include='number').columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# 2. Check distribution of transactions amount in the data
plt.figure(figsize=(8, 6))
sns.histplot(data['Transaction_Amount'], kde=False)
plt.title('Distribution of Transaction Amount')
plt.show()

# 3. Check distribution of transactions amount by account type
plt.figure(figsize=(8, 6))
sns.boxplot(x='Account_Type', y='Transaction_Amount', data=data)
plt.title('Transaction Amount by Account Type')
plt.show()

# 4. Check the average transaction amount by age.
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Average_Transaction_Amount', hue='Account_Type', data=data)
plt.title('Average Transaction Amount vs. Age')
plt.show()

# 5. Check the count of transactions by day of the week
plt.figure(figsize=(8, 6))
sns.countplot(x='Day_of_Week', data=data)
plt.title('Count of Transactions by Day of the Week')
plt.xticks(rotation=45)
plt.show()

# 6. Check the correlation between all the numeric columns in the data
numeric_data = data[numeric_columns]
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='viridis')
plt.title('Correlation Heatmap')
plt.show()

# Select the relevant features
features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']
X = data[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. Calculate the number of anomalies in the data to find the ratio of anomalies in the data,
# which will be useful while using anomaly detection algorithms like isolation forest.
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_scaled)
y_pred = model.predict(X_scaled)
anomaly_count = (y_pred == -1).sum()
total_count = len(data)
anomaly_ratio = anomaly_count / total_count
print(f'Number of anomalies: {anomaly_count}')
print(f'Total number of transactions: {total_count}')
print(f'Anomaly ratio: {anomaly_ratio}')

# Create the 'Is_Anomaly' column
data['Is_Anomaly'] = (y_pred == -1).astype(int)

# 7. Visualize anomalies in the data
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Transaction_Amount', y='Average_Transaction_Amount', hue='Is_Anomaly', data=data)
plt.title('Anomalies in Transaction Amount')
plt.show()

# 9. Select the relevant features and fit them into the Machine Learning model “isolation forest” 
# for detecting anomalies. Now get the prediction and convert into binary values.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, data['Is_Anomaly'], test_size=0.2, random_state=42)

model = IsolationForest(contamination=anomaly_ratio, random_state=42)
model.fit(X_train)
y_pred = model.predict(X_test)
y_pred = (y_pred == -1).astype(int)

# 10. Show the classification report
print(classification_report(y_test, y_pred))

# 11. Use the trained model to detect anomalies to bring following result.
# Enter the value for 'Transaction_Amount': 10000
# Enter the value for 'Average_Transaction_Amount': 900
# Enter the value for 'Frequency_of_Transactions': 6
# Anomaly detected: This transaction is flagged as an anomaly.

# Create a new data point to test
new_transaction = pd.DataFrame({
    'Transaction_Amount': [10000],
    'Average_Transaction_Amount': [900],
    'Frequency_of_Transactions': [6]
})

# Scale the new data point using the same scaler as before
new_transaction_scaled = scaler.transform(new_transaction)

# Predict the anomaly score for the new data point
anomaly_score = model.predict(new_transaction_scaled)

# Check if the transaction is an anomaly
if anomaly_score[0] == -1:
    print('Anomaly detected: This transaction is flagged as an anomaly.')
else:
    print('This transaction is not flagged as an anomaly.')