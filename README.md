# Basic-Data-Analysis-Code-Python-
Python Data Analysis Code
# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Data
data = pd.read_csv('data.csv')

# Step 3: Data Cleaning
# Handle missing values
data = data.dropna()  # Drop rows with missing values
# data.fillna(method='ffill', inplace=True)  # Alternatively, fill missing values

# Check for duplicates
data = data.drop_duplicates()

# Step 4: Exploratory Data Analysis (EDA)
print(data.describe())
print(data.info())

# Correlation matrix
corr_matrix = data.corr()
print(corr_matrix)

# Step 5: Data Visualization
# Histograms
data.hist(bins=30, figsize=(20, 15))
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Scatter plot
sns.pairplot(data)
plt.show()

# Step 6: Feature Engineering
# Example: Creating a new feature
data['new_feature'] = data['existing_feature1'] * data['existing_feature2']

# Step 7: Model Building
# Define features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Model Evaluation
# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Plotting true vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions')
plt.show()
