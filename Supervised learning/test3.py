# 📦 Import essential libraries for data processing, modeling, and visualization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# 📂 Load the dataset
# The dataset contains two columns: Time (in seconds) and Heart Rate (in BPM)
hr = pd.read_csv(r'G:\My Drive\تاهيل وتدريب\الذكاء الاصطناعي\Model training data\AI Files\AI Files\2 - Polynomial regression\heart_rate.txt')

# 🧮 Extract the input features (X) and target variable (y)
# X -> Time values | y -> Corresponding Heart Rate values
x = hr.iloc[:, :-1].values
y = hr.iloc[:, 1].values

# 🔀 Split the data into training and test sets (90% training, 10% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

# ⚙️ Train a simple Linear Regression model (as a baseline)
lin = LinearRegression()
lin.fit(x_train, y_train)

# 🔁 Transform input features into polynomial features (degree = 4)
# This helps the model capture non-linear relationships in the data
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(x_train)

# 🧠 Train the Polynomial Regression model using the transformed features
lin2 = LinearRegression()
lin2.fit(X_poly, y_train)

# 🔍 Predict target values for the test set using the polynomial model
y_pred = lin2.predict(poly.transform(x_test))

# 📊 Visualize the original data points and the fitted polynomial curve
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Original Data')  # Raw data points
plt.plot(x, lin2.predict(poly.transform(x)), color='red', label='Polynomial Fit')  # Fitted curve
plt.title('Polynomial Regression Fit (Degree 4)')
plt.xlabel('Time (seconds)')
plt.ylabel('Heart Rate (BPM)')
plt.legend()

# 📋 Create a DataFrame to compare actual vs predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
print(df1)

# 📈 Plot a bar chart for a visual comparison of actual vs predicted heart rates
df1.plot(kind='bar', figsize=(10, 6))
plt.title('Actual vs Predicted Heart Rate (Top 25 Samples)')
plt.xlabel('Sample Index')
plt.ylabel('Heart Rate (BPM)')
plt.tight_layout()

# 🧾 Print model performance metrics to evaluate accuracy
print('📉 Mean Absolute Error (MAE):', mean_absolute_error(y_test, y_pred))
print('📉 Mean Squared Error (MSE):', mean_squared_error(y_test, y_pred))
print('📉 Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error(y_test, y_pred)))
print('📈 R² Score:', r2_score(y_test, y_pred))  # How well the model explains the variability of the data

# 🖼️ Show all plots
plt.show()
