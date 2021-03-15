"""
Task 1a
Team Naiveoutliers
Robin Schmid, Pascal Müller, Marvin Harms
Mar, 2021

This model performs a Ridge regression for different lambda parameters
and computes the RMSE for each lambda using cross validation with 10 folds.
Use RepeatedKFold to average over multiple random seeds.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold

# Parameters
n_folds = 10
lambdas = [0.1, 1, 10, 100, 200]

# Load data
train_file = pd.read_csv("../handout/train.csv")

y_data = train_file.values[:,0]
x_data = train_file.values[:,1:]

n_repeats = 200
rkf = RepeatedKFold(n_splits=n_folds, random_state=0, n_repeats=n_repeats) # Creates different splits at each repetition, fixed seed fixes splits in each iteration
i = 0
RMSE = np.zeros(len(lambdas))
for l in lambdas:
    RMSE_sum = 0.0

    for train_index, test_index in rkf.split(x_data):

        x_train = x_data[train_index]
        y_train = y_data[train_index]

        x_test = x_data[test_index]
        y_test = y_data[test_index]

        model = Ridge(alpha=l)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        RMSE_sum += mean_squared_error(y_test, y_pred)**0.5

    RMSE[i] = RMSE_sum/(n_folds*n_repeats)  # Average RMSE over all folds
    i = i + 1

np.savetxt("submission3.csv", RMSE, comments='', delimiter=",", fmt="%s")