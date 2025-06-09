# === Import Required Libraries === #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# === Load Dataset === #
data_path = r'G:\My Drive\تاهيل وتدريب\الذكاء الاصطناعي\Model training data\AI Files\AI Files\3 - Logistic regression\diabetes.csv'
df = pd.read_csv(data_path)
print("First 5 rows of the dataset:")
print(df.head())

# === Define Features and Target === #
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[features]
y = df['Outcome']

# === Split the Dataset === #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# === Create and Train the Model === #
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# === Make Predictions === #
y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]

# === Evaluate the Model === #
accuracy = accuracy_score(y_test, y_pred_test)
roc_auc = roc_auc_score(y_test, y_proba_test)
conf_matrix = confusion_matrix(y_test, y_pred_test)

# === Display Evaluation Metrics === #
print(f"\nTest Accuracy     : {accuracy * 100:.2f}%")
print(f"ROC AUC Score     : {roc_auc * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

# === Plot Confusion Matrix === #
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# === Plot ROC Curve === #
fpr, tpr, thresholds = roc_curve(y_test, y_proba_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
