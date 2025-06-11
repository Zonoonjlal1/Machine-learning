# import 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
# Load dataset from CSV file
Dataset = pd.read_csv(r'G:\My Drive\تاهيل وتدريب\الذكاء الاصطناعي\Model training data\AI Files\AI Files\2 - Polynomial regression\data.csv')
# Display the first 5 rows of the dataset
x = Dataset.iloc[:, 1:2].values
y = Dataset.iloc[:, 2].values
x_train, Xtest, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# Train a simple Linear Regression model
liner = LinearRegression()
liner.fit(x_train, y_train)
# Generate polynomial features (degree 4)
poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(x_train)  # Transform training features to polynomial features
# Train Polynomial Regression model using transformed features
liner2 = LinearRegression()
liner2.fit(x_poly,y_train )

plt.scatter(x, y, color='blue', label='Original Data')
plt.plot(x, liner2.predict(poly.fit_transform(x)), color='red', label='Polynomial Regression')
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.legend()


# Predict using the simple linear model (optional/for comparison)
y_prod = liner.predict(x_train)
#print("Predicted values using Linear Regression:", y_prod)

zain = 110
zain = np.array([[zain]])
test = liner2.predict(poly.fit_transform(zain))
print(test)
#print("Predicted value for zain (200) using Polynomial Regression:", test)
detcted = PolynomialFeatures(4)
detcted_lin = LinearRegression()
detcted_lin.fit(detcted.fit_transform(x),y)
dx = detcted_lin.predict(detcted.fit_transform(x))
print(mean_squared_error(y,dx))
plt.show()
