# import module
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import dataset
my_file = r'G:\My Drive\تاهيل وتدريب\الذكاء الاصطناعي\Model training data\student_study_data.csv'
Dataset = pd.read_csv(my_file)
# check th Dataset 
#print(Dataset.head())

# visualize the Dataset
Dataset.plot(x='Hours',y='Scores', style='o')
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')  
#plt.show()

# Splitting the dataset into features and target variable
x = Dataset.iloc[: , :-1].values
y = Dataset.iloc[: ,   1].values

# Splitting the dataset into training and testing sets
regesor = LinearRegression()
x_train, x_test, y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# Training the model
regesor.fit(x_train,y_train)
# Making predictions on the test set
y_pred = regesor.predict(x_test)   
df=pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
# splitting
df1 = df.head(10)
print(df1)
df1.plot(kind='bar',figsize=(10,6))
plt.grid(which='major', linestyle='--', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
# Evaluating the model
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R-squared Score (Accuracy):', r2_score(y_test, y_pred))
# Visualizing the regression line
plt.scatter(x, y, color='blue')
plt.plot(x_train, regesor.predict(x_train), color='red')
plt.title('Regression Line')    
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()