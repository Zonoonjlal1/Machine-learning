# ===============================
# 📦 Importing Required Libraries
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

# =======================================
# 📂 Load Dataset from Local File System
# =======================================
path_of_data = r'G:\My Drive\تاهيل وتدريب\الذكاء الاصطناعي\Model training data\AI Files\AI Files\3 - Logistic regression\diabetes.csv'
dataset = pd.read_csv(path_of_data)

# ============================================
# 🧪 Define Features (X) and Target Variable (y)
# ============================================
X = dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = dataset['Outcome']

# ======================================
# 🔀 Split the Data into Train and Test
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0
)

# ==========================================
# 🧠 Initialize and Train Logistic Regression
# ==========================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =================================
# 🔍 Make Predictions on Test Data
# =================================
y_pred = model.predict(X_test)                    # Class predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability estimates

# ===============================
# 📊 Evaluate Model Performance
# ===============================
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Display metrics
print(f"\n✅ Test Accuracy     : {accuracy * 100:.2f}%")
print(f"✅ ROC AUC Score     : {roc_auc * 100:.2f}%")

# ===================================
# 📈 Bar Plot: Actual vs. Predicted
# ===================================
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
results_df.head(25).plot(kind='bar', figsize=(10, 6))
plt.title('Actual vs Predicted Outcomes')
plt.xlabel('Test Sample Index')
plt.ylabel('Outcome')
plt.tight_layout()
plt.show()

# ===============================

# 🔥 Visualize Confusion Matrix
# ===============================
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# ========================
# 📉 Plot the ROC Curve
# ========================
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
