# ====================================================
# 📦 Import Required Libraries
# ====================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ====================================================
# 📁 Load the Dataset
# ====================================================
my_file = r'G:\My Drive\تاهيل وتدريب\الذكاء الاصطناعي\Model training data\student_study_data.csv'
Dataset = pd.read_csv(my_file)

# ====================================================
# 📊 Visualize the Dataset (Scatter Plot)
# ====================================================
Dataset.plot(x='Hours', y='Scores', style='o')
plt.title('Hours Studied vs Percentage Score')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
# plt.show()  # Uncomment to display the plot

# ====================================================
# 🧮 Prepare Features and Target Variable
# ====================================================
x = Dataset.iloc[:, :-1].values     # Features (Hours)
y = Dataset.iloc[:, 1].values       # Target (Scores)

# ====================================================
# 🧪 Split Data into Training and Testing Sets
# ====================================================
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# ====================================================
# 🧠 Train the Linear Regression Model
# ====================================================
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# ====================================================
# 📈 Make Predictions on the Test Set
# ====================================================
y_pred = regressor.predict(x_test)

# ====================================================
# 📋 Compare Actual vs Predicted Results
# ====================================================
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)

# Show top 10 results
df1 = df.head(10)
print(df1)

# ====================================================
# 📊 Visualize Actual vs Predicted using Bar Chart
# ====================================================
df1.plot(kind='bar', figsize=(10, 6))
plt.grid(which='major', linestyle='--', linewidth=0.5, color='green')
plt.grid(which='minor', linestyle=':', linewidth=0.5, color='black')
plt.title('Actual vs Predicted Scores')
plt.xlabel('Test Cases')
plt.ylabel('Scores')
plt.show()

# ====================================================
# 🧾 Evaluate the Model Performance
# ====================================================
print('📌 Mean Absolute Error       :', mean_absolute_error(y_test, y_pred))
print('📌 Mean Squared Error       :', mean_squared_error(y_test, y_pred))
print('📌 Root Mean Squared Error  :', np.sqrt(mean_squared_error(y_test, y_pred))
)
print('📌 R-squared Score (Accuracy):', r2_score(y_test, y_pred))

# ====================================================
# 🧩 Visualize Regression Line
# ====================================================
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x_train, regressor.predict(x_train), color='red', label='Regression Line')
plt.title('Regression Line: Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.legend()
plt.show()
