# ============================================================
#                 Importing Required Libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error

# ============================================================
#                 Loading and Exploring the Dataset
# ============================================================
dataset = pd.read_csv(r'G:\My Drive\تاهيل وتدريب\الذكاء الاصطناعي\Model training data\student_scores.csv')

# Preview the first and last few rows of the dataset
print(dataset.head())
print(dataset.tail())

# Display the dataset's shape, structure, and statistical summary
print(dataset.shape)
print(dataset.info())
print(dataset.describe().sum())
print(dataset.describe())

# ============================================================
#                 Plotting the Data
# ============================================================
# Visualize the relationship between study hours and scores
dataset.plot(
    x='Hours', 
    y='Scores', 
    style='o', 
    title='Hours vs Scores',
    xlabel='Hours Studied',
    ylabel='Percentage Score',
    color='blue',
    linewidth=2
)
plt.show()

# ============================================================
#                 Preparing the Data
# ============================================================
# Separating the feature (Hours) and the target variable (Scores)
x = dataset.iloc[:, :-1].values
print(x)
y = dataset.iloc[:, 1].values
print(y)

# Splitting the dataset into training (80%) and testing (20%) sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# ============================================================
#                 Training the Linear Regression Model
# ============================================================
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# ============================================================
#                 Making Predictions
# ============================================================
y_prod = regressor.predict(x_test)  # Predicted scores for test set

# Creating a DataFrame to compare actual vs predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_prod})
print(df)

# ============================================================
#                 Visualizing the Predictions
# ============================================================
df1 = df.head(25)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='--', color='green')
plt.grid(which='minor', linestyle=':', color='black')
plt.show()

# ============================================================
#                 Evaluating the Model
# ============================================================
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_prod))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_prod))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_prod)))
print('R-squared Score (Accuracy):', regressor.score(x, y))
