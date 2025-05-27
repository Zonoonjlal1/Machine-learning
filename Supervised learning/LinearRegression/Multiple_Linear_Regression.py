# ====================================================
# 📦 Import Required Libraries
# ====================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# ====================================================
# 📁 Load the Dataset
# ====================================================
my_file = r'G:\My Drive\تاهيل وتدريب\الذكاء الاصطناعي\Model training data\AI Files\AI Files\1 - Linear regression (simple , multi)\multi\petrol_consumption.csv'
Dataset = pd.read_csv(my_file)
print(Dataset.head())

# ====================================================
# 🧮 Define Features and Target Variable
# ====================================================
x = Dataset[['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)']]
y = Dataset['Petrol_Consumption']

# ====================================================
# 🧠 Initialize and Train the Linear Regression Model
# ====================================================
regression = LinearRegression()
regression.fit(x, y)

# ====================================================
# 🧪 Split the Dataset into Training and Testing Sets
# ====================================================
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# Retrain the model on training data
regression.fit(x_train, y_train)

# ====================================================
# 📈 Make Predictions on the Test Set
# ====================================================
y_pred = regression.predict(x_test)
print("🔹 Predicted values:\n", y_pred)

# ====================================================
# 📋 Compare Actual vs Predicted Values
# ====================================================
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)

# Plot top 15 results for better visualization
df1 = df.head(15)
df1.plot(kind='bar', figsize=(10, 6))
plt.grid(which='major', linestyle='--', linewidth=0.5, color='green')
plt.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
plt.title('Actual vs Predicted Petrol Consumption')
plt.xlabel('Test Cases')
plt.ylabel('Petrol Consumption')
plt.tight_layout()
plt.show()

# ====================================================
# 🧾 Evaluate the Model Performance
# ====================================================
print('📌 Mean Absolute Error       :', mean_absolute_error(y_test, y_pred))
print('📌 Mean Squared Error        :', mean_squared_error(y_test, y_pred))
print('📌 Root Mean Squared Error   :', np.sqrt(mean_squared_error(y_test, y_pred)))
print('📌 R-squared Score (Accuracy):', r2_score(y_test, y_pred) * 100, '%')

# Alternatively, using model’s built-in score method
print('📌 Coefficient of Determination (score):', regression.score(x_test, y_test))
