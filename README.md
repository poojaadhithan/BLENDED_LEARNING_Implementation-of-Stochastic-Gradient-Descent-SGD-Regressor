# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and preprocess the data by removing unnecessary columns and converting categorical variables.
2. Split the dataset into features and target variable, then standardize the data and divide it into training and testing sets.
3. Train the SGD Regressor model using the training data.
4. Predict the test data and evaluate the model using MSE, MAE, and R², and visualize the results.

## Program:
```
#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import pandas as pd
data = pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())

#Data preprocessing
#Dropping unnecessary columns and handling categorical variables
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)

#Splitting the data info features and target variable
X = data.drop('price', axis=1)
y = data['price']

#Standardizing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1, 1))

#Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creating the SGD Regressor model
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)

#Fitting the model on the training data
sgd_model.fit(X_train, y_train.ravel())

#Making predictions
y_pred = sgd_model.predict(X_test)

#Evaluating model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("="*50)
print('Name: POOJA A')
print('Reg No: 212225040300')
print(f"MSE: {mse:.4f}")
print(f"R2: {r2_score(y_test,y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.4f}")
print("="*50)

#Print model coefficients
print("Model coefficients:")
print("coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)

#Visualizing actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGDRegressor")
plt.plot([min(y_test), max(y_test)],
         [min(y_test), max(y_test)],
         color='red')
plt.show()
```

## Output:
<img width="316" height="114" alt="image" src="https://github.com/user-attachments/assets/0ce1e010-8c5e-42b9-a007-05494f72650d" />
<img width="767" height="551" alt="image" src="https://github.com/user-attachments/assets/c138fdf3-57c9-43d5-955d-feca6cb4dfd3" />





## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
