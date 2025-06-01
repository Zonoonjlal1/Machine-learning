# import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# Load dataset from CSV file
Dataset = pd.read_csv(r'G:\My Drive\تاهيل وتدريب\الذكاء الاصطناعي\Model training data\AI Files\AI Files\1 - Linear regression (simple , multi)\student\student-mat.csv', sep=';')
# print the first 5 rows of the dataset
print(Dataset.head())

x = Dataset[['G1','G2','G3','studytime', 'failures', 'absences']].values
y = Dataset['G3'].values
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, random_state=0)
# Train a simple Linear Regression model
lin = LinearRegression()
lin.fit(x_train, y_train)
# Predict using the simple linear model
y_brod = lin.predict(x_test)
# print("Predicted values using Linear Regression:", y_brod)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_brod})
print(df)
# Plot top 15 results for better visualization
df1 = df.head(15)
df1.plot(kind='bar', figsize=(10, 6))
plt.grid(which='major', linestyle='--', linewidth=0.5, color='green')
plt.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
plt.show()
# Calculate and print evaluation metrics
print("Mean Absolute Error:", mean_absolute_error(y_test, y_brod))
print("Mean Squared Error:", mean_squared_error(y_test, y_brod))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_brod)))
print("R2 Score:", r2_score(y_test, y_brod))