"""
Introduction to Machine Learning
Task 1b: Linear regression of feature transformations
Submission NLT021200BAPR21
Team Naiveoutliers
March 2021

Description of this file
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


# load data
data = pd.read_csv("../handout/train.csv")

# split labels y and features x
y_full = data.values[:, 1]
X_full = data.values[:, 2:]

# create feature transformation matrix from 700x5 to 700x21
X_feature = np.hstack((X_full,  # linear
                       X_full**2,  # quadratic
                       np.exp(X_full),  # exponential
                       np.cos(X_full),  # cosine
                       np.ones((X_full.shape[0], 1))))  # constant

# perform evaluation with RMSE and KFold and print computed RMSE
rmse = 0.0
n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(X=X_feature, y=y_full):
    model = LinearRegression(fit_intercept=False)
    model.fit(X=X_feature[train_idx], y=y_full[train_idx])
    y_pred = model.predict(X=X_feature[test_idx])
    rmse += mean_squared_error(y_full[test_idx], y_pred) ** 0.5
print(rmse/n_folds)

# perform linear regression on feature matrix
model = LinearRegression(fit_intercept=False)  # accepts intercepts not only through origin
model.fit(X=X_feature, y=y_full)

# save the weights to .csv file
np.savetxt('weights.csv', model.coef_, fmt='%s')



