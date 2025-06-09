# =========================
# Import Required Libraries
# =========================
import numpy as np                                # For numerical computations
import pandas as pd                               # For data manipulation
import matplotlib.pyplot as plt                   # For data visualization
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.linear_model import LinearRegression     # For linear regression modeling
from sklearn.metrics import mean_squared_error, r2_score  # For model evaluation metrics
from sklearn import metrics                            # Additional metrics
from sklearn.preprocessing import PolynomialFeatures   # To generate polynomial features

# ======================
# Load and Inspect Data
# ======================
# Read dataset from CSV file
Dataset = pd.read_csv(r'G:\My Drive\تاهيل وتدريب\الذكاء الاصطناعي\Model training data\AI Files\AI Files\2 - Polynomial regression\data.csv')

# Display the first 5 rows of the dataset
print(Dataset.head())

# ===============================
# Split Dataset into X and Y
# ===============================
# Extract independent variable (e.g., Temperature) and dependent variable (e.g., Pressure)
x = Dataset.iloc[:, 1:2].values  # Selects the second column as features
y = Dataset.iloc[:, 2].values    # Selects the third column as target

# Split dataset into training and testing sets (80% train, 20% test)
x_train, Xtest, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# ===============================
# Train Linear and Polynomial Models
# ===============================

# Train a simple Linear Regression model
lin = LinearRegression()
lin.fit(x_train, y_train)

# Generate polynomial features (degree 4)
poly = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(x_train)  # Transform training features to polynomial features

# Train Polynomial Regression model using transformed features
lin2 = LinearRegression()
lin2.fit(x_poly, y_train)

# Predict using the simple linear model (optional/for comparison)
y_prod = lin.predict(x_train)

# ===============================
# Visualization
# ===============================

# Plot original data points
plt.scatter(x, y, color='blue', label='Original Data')

# Plot Polynomial Regression Line
plt.plot(x, lin2.predict(poly.fit_transform(x)), color='red', label='Polynomial Regression')

# Plot training data separately for clarity
plt.scatter(x_train, y_train, color='y', label='Training Data')

# Add titles and labels
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.legend()

# Uncomment to display the plot
plt.show()

# ===============================
# Model Evaluation on Test Set
# ===============================

# Evaluate model performance using test data
print("Mean Squared Error:", mean_squared_error(y_test, lin2.predict(poly.fit_transform(Xtest))))
print("R-squared:", r2_score(y_test, lin2.predict(poly.fit_transform(Xtest))))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, lin2.predict(poly.fit_transform(Xtest)))))
