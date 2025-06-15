# ===============================
# 📦 Import Required Libraries
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# =======================================
# 📂 Load Dataset from Local CSV File
# =======================================
data_path = r'G:\My Drive\تاهيل وتدريب\الذكاء الاصطناعي\Model training data\AI Files\AI Files\4 - Naive Bayes\NaiveBayes.csv'
dataset = pd.read_csv(data_path)

# ============================================
# ✂️ Extract Features (X) and Target Variable (y)
# ============================================
X = dataset.iloc[:, [0, 1]].values   # Selecting first and second columns as features
y = dataset.iloc[:, 2].values        # Third column is the target

# ====================================================
# 🔀 Split Data into Training and Testing Sets (90/10)
# ====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0
)

# ============================================
# 🧠 Train Bernoulli Naive Bayes Classifier
# ============================================
bernoulli_model = BernoulliNB()
bernoulli_model.fit(X_train, y_train)

# 📈 Make Predictions and Evaluate Accuracy
y_pred_bernoulli = bernoulli_model.predict(X_test)
acc_bernoulli = accuracy_score(y_test, y_pred_bernoulli)
print(f"✅ BernoulliNB Accuracy: {acc_bernoulli * 100:.2f}%")

# ============================================
# 🧠 Train Gaussian Naive Bayes Classifier
# ============================================
gaussian_model = GaussianNB()
gaussian_model.fit(X_train, y_train)

# 📈 Make Predictions and Evaluate Accuracy
y_pred_gaussian = gaussian_model.predict(X_test)
acc_gaussian = accuracy_score(y_test, y_pred_gaussian)
print(f"✅ GaussianNB Accuracy: {acc_gaussian * 100:.2f}%")

# ============================================
# 📊 Confusion Matrix: Bernoulli Naive Bayes
# ============================================
cm_bernoulli = confusion_matrix(y_test, y_pred_bernoulli)
sns.heatmap(cm_bernoulli, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - BernoulliNB')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ============================================
# 📊 Confusion Matrix: Gaussian Naive Bayes
# ============================================
cm_gaussian = confusion_matrix(y_test, y_pred_gaussian)
sns.heatmap(cm_gaussian, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - GaussianNB')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('model3.png')  # Save the figure
plt.show()
