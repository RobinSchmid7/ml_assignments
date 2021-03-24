"""
Task 1b
Team Naiveoutliers
Robin Schmid, Pascal Müller, Marvin Harms
Mar, 2021

Use feature transformations and linear regression to fit the model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

evaluation = False # Evaluation with RepeatedKFold

full_data = pd.read_csv("../handout/train.csv")

y_data = full_data.values[:,1]
x_data = full_data.values[:,2:]

# Perform feature transformation
x_features = np.hstack((x_data, x_data**2, np.exp(x_data), np.cos(x_data), np.ones((x_data.shape[0],1))))

# Fit model
model = LinearRegression(fit_intercept=False) # Allow all linear functions not just the ones fitting through the origin
model.fit(x_features, y_data)

# Save weights
np.savetxt("submission.csv",model.coef_,comments='',delimiter=",",fmt="%s")

# Evaluation with Repeated KFold
if evaluation:
    n_folds = 20
    n_repeats = 100
    RMSE_sum = 0

    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats)

    for train_index, test_index in rkf.split(x_features):

        x_train = x_features[train_index]
        y_train = y_data[train_index]

        x_test = x_features[test_index]
        y_test = y_data[test_index]

        model = LinearRegression()
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        RMSE_sum += mean_squared_error(y_test, y_pred) ** 0.5

    RMSE = RMSE_sum / (n_folds * n_repeats)  # Average RMSE over all folds
    print(RMSE)