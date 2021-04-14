"""
Introduction to Machine Learning
Task 1b: Linear regression of feature transformations
Team Naiveoutliers
March 2021

Description of this file
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold

# parameters for ridge regression
ridge_alpha = np.linspace(0.001, 0.1, 10000)

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

# perform evaluation with RMSE and KFold
rmse_kf = 0.0
n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(X=X_feature, y=y_full):
    model = RidgeCV(alphas=ridge_alpha, fit_intercept=False, store_cv_values=True)
    # model = LinearRegression(fit_intercept=False)
    model.fit(X=X_feature[train_idx], y=y_full[train_idx])
    y_pred = model.predict(X=X_feature[test_idx])
    rmse_kf += mean_squared_error(y_full[test_idx], y_pred) ** 0.5
print("RMSE 10-fold:", rmse_kf/n_folds)

# perform evaluation with RMSE and repeated KFold
rmse_rkf = 0.0
n_repeats = 10
rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
for train_idx, test_idx in rkf.split(X=X_feature, y=y_full):
    model = RidgeCV(alphas=ridge_alpha, fit_intercept=False, store_cv_values=True)
    # model = LinearRegression(fit_intercept=False)
    model.fit(X=X_feature[train_idx], y=y_full[train_idx])
    y_pred = model.predict(X=X_feature[test_idx])
    rmse_rkf += mean_squared_error(y_full[test_idx], y_pred) ** 0.5
print("RMSE repeated 10-fold:", rmse_rkf / (n_folds * n_repeats))

# perform linear regression on feature matrix
lin_model = LinearRegression(fit_intercept=False)  # accepts intercepts not only through origin
lin_model.fit(X=X_feature, y=y_full)

# perform ridge regression of feature matrix
ridge_model = RidgeCV(alphas=ridge_alpha, fit_intercept=False, store_cv_values=True)
ridge_model.fit(X=X_feature, y=y_full)

# save the weights to .csv file
np.savetxt('linear.csv', lin_model.coef_, fmt='%s')
np.savetxt('ridge.csv', ridge_model.coef_, fmt='%s')
