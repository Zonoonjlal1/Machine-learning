# ============================================================
#                  Import Required Libraries
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================
#                  Load and Prepare the Dataset
# ============================================================

# Define file path and columns of interest
file_path = r'E:\Machine learning\Supervised learning\LinearRegression\Model training data\Real Estate Sales 2001-2022.csv'
columns_needed = ['Assessed Value', 'Sale Amount']

# Read dataset and filter relevant columns
Dataset = pd.read_csv(file_path)
Dataset = Dataset[columns_needed].copy()

# ============================================================
#                  Data Cleaning: Outlier Removal
# ============================================================

# Remove outliers using the Interquartile Range (IQR) method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Remove outliers using the Z-Score method
def remove_outliers_zscore(df, threshold=3):
    z_scores = np.abs(zscore(df))
    return df[(z_scores < threshold).all(axis=1)]

# Apply outlier removal iteratively for better results
def clean_data_iteratively(df, columns, max_iter=10):
    prev_shape = None
    iter_count = 0
    while prev_shape != df.shape and iter_count < max_iter:
        prev_shape = df.shape
        for col in columns:
            df = remove_outliers_iqr(df, col)
        df = remove_outliers_zscore(df)
        iter_count += 1
    return df

# Clean the dataset
Dataset_clean = clean_data_iteratively(Dataset, columns_needed)

# ============================================================
#                  Data Visualization (Scatter Plot)
# ============================================================

Dataset_clean.plot(
    x='Assessed Value',
    y='Sale Amount',
    style='o',
    title='Assessed Value vs Sale Price',
    xlabel='Assessed Value',
    ylabel='Sale Price',
    color='blue',
    linewidth=2
)
plt.show()

# ============================================================
#                  Data Preparation for Modeling
# ============================================================

# Define feature (X) and target (y)
x = Dataset_clean.iloc[:, :-1].values   # 'Assessed Value'
y = Dataset_clean.iloc[:, 1].values     # 'Sale Amount'

# Split dataset into training and test sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# ============================================================
#                  Train Linear Regression Model
# ============================================================

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict on the test data
y_pred = regressor.predict(x_test)

# Compare actual vs predicted values
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# ============================================================
#                  Visualization: Prediction Results
# ============================================================

df1.head(25).plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='--', color='green')
plt.grid(which='minor', linestyle=':', color='black')
plt.show()

# ============================================================
#                  Model Evaluation Metrics
# ============================================================

print('Mean Absolute Error (MAE):', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error (MSE):', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R-squared (Model Accuracy):', regressor.score(x, y))  # Percentage of variance explained
