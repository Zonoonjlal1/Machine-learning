import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# ---------------------------------------
# Step 1: Load the dataset
# ---------------------------------------
Dataset = pd.read_csv(
    r'G:\My Drive\تاهيل وتدريب\الذكاء الاصطناعي\Model training data\AI Files\AI Files\5 - Decision Tree classifier\diabetes.csv'
)

# Display the first 5 rows to get a glimpse of the data
print("Dataset Preview:")
print(Dataset.head())

# Show the dataset dimensions (rows, columns)
print("\nDataset Shape:")
print(Dataset.shape)

# Show statistical summary for numerical columns
print("\nStatistical Summary:")
print(Dataset.describe())

# Display concise information about the dataset, data types and null values
print("\nDataset Info:")
print(Dataset.info())

# Check for duplicate rows by counting unique occurrences of entire rows
print("\nValue Counts (to check duplicates):")
print(Dataset.value_counts())

# List all column names for reference
print("\nColumn Names:")
print(Dataset.columns)

# ---------------------------------------
# Step 2: Define features and target variable
# ---------------------------------------
features = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

X = Dataset[features]  # Feature matrix
y = Dataset['Outcome']  # Target vector

# ---------------------------------------
# Step 3: Split the dataset into training and testing sets
# ---------------------------------------
# 90% training data, 10% testing data
# random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# ---------------------------------------
# Step 4: Initialize and train the Decision Tree model
# ---------------------------------------
model = DecisionTreeClassifier()

# Fit the model on the training data
model.fit(X_train, y_train)

# ---------------------------------------
# Step 5: Make predictions
# ---------------------------------------
y_pred_test = model.predict(X_test)    # Predictions on test data
y_pred_train = model.predict(X_train)  # Predictions on training data

# ---------------------------------------
# Step 6: Evaluate the model accuracy
# ---------------------------------------
test_accuracy = accuracy_score(y_test, y_pred_test)
train_accuracy = accuracy_score(y_train, y_pred_train)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Train Accuracy: {train_accuracy:.4f}")

# ---------------------------------------
# Step 7: Export and display the decision tree rules
# ---------------------------------------
tree_rules = tree.export_text(model)
print("\nDecision Tree Rules:")
print(tree_rules)
