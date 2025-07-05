# --- Libraries Import ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

# --- Dataset Loading ---
Dataset = pd.read_csv(r'G:\My Drive\تاهيل وتدريب\الذكاء الاصطناعي\Model training data\AI Files\AI Files\6 - SVM\customer_purchases.csv')

# --- Feature (X) and Target (y) Slicing ---
x = Dataset.iloc[:, [0, 1]].values
y = Dataset.iloc[:, -1].values

# --- Data Splitting (Train/Test) ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

# --- Feature Scaling ---
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# --- Linear SVM Model Training ---
model_linear = SVC(kernel='linear', random_state=0)
model_linear.fit(x_train, y_train)
y_pred_linear = model_linear.predict(x_test)
print(f"Accuracy with Linear Kernel: {accuracy_score(y_test, y_pred_linear):.4f}")

# --- Decision Boundary Plotting Function ---
def plot_decision_boundary(X_set, y_set, classifier, title, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('black', 'white')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'blue'))(i), label=f'Class {j}', edgecolors='k', s=50)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# --- Linear Kernel Visualization (Training Set) ---
plot_decision_boundary(x_train, y_train, model_linear,
                       'SVM (Linear Kernel) - Training Set: Purchased vs Non-Purchased', 'Scaled Salary', 'Scaled Age')

# --- Linear Kernel Visualization (Test Set) ---
plot_decision_boundary(x_test, y_test, model_linear,
                       'SVM (Linear Kernel) - Test Set: Purchased vs Non-Purchased', 'Scaled Salary', 'Scaled Age')

# --- RBF SVM Model Training ---
model_rbf = SVC(kernel='rbf', random_state=0)
model_rbf.fit(x_train, y_train)
y_pred_rbf = model_rbf.predict(x_test)
print(f"Accuracy with RBF Kernel: {accuracy_score(y_test, y_pred_rbf):.4f}")

# --- RBF Kernel Visualization (Training Set) ---
plot_decision_boundary(x_train, y_train, model_rbf,
                       'SVM (RBF Kernel) - Training Set: Purchased vs Non-Purchased', 'Scaled Salary', 'Scaled Age')

# --- RBF Kernel Visualization (Test Set) ---
plot_decision_boundary(x_test, y_test, model_rbf,
                       'SVM (RBF Kernel) - Test Set: Purchased vs Non-Purchased', 'Scaled Salary', 'Scaled Age')

# --- Confusion Matrix (Linear Kernel) ---
cm_linear = confusion_matrix(y_test, y_pred_linear)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Blues',
            xticklabels=model_linear.classes_, yticklabels=model_linear.classes_)
plt.title('Confusion Matrix - Linear Kernel')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# --- Confusion Matrix (RBF Kernel) ---
cm_rbf = confusion_matrix(y_test, y_pred_rbf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Greens',
            xticklabels=model_rbf.classes_, yticklabels=model_rbf.classes_)
plt.title('Confusion Matrix - RBF Kernel')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()