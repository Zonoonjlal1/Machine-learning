#import 
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
# data path
path_of_data = r'G:\My Drive\تاهيل وتدريب\الذكاء الاصطناعي\Model training data\AI Files\AI Files\3 - Logistic regression\diabetes.csv'
# load data
Data = pd.read_csv(path_of_data)
print(Data.head())
# define features and target variable
X = Data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = Data['Outcome']
# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1,random_state=0)
# initialize and train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# make predictions on test data
y_pred = model.predict(X_test)                    # Class predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability estimates
# evaluate model performance
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n✅ Test Accuracy     : {accuracy * 100:.2f}%")
print(f"✅ ROC AUC Score     : {roc_auc * 100:.2f}%")
print(f"Confusion Matrix:\n{conf_matrix}")
# roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid()
plt.show()
# confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
# bar plot: actual vs predicted
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
results_df.head(25).plot(kind='bar', figsize=(10, 6))
plt.title('Actual vs Predicted Outcomes')
plt.xlabel('Test Sample Index')
plt.ylabel('Outcome')
plt.tight_layout()
plt.show()
