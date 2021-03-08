"""
Task 1a
Team Naiveoutliers
Robin Schmid, Pascal Müller, Marvin Harms
Mar, 2021
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Parameters
n_folds = 10
lambdas = [0.1, 1, 10, 100, 200]

# Load data
train_file = pd.read_csv("../handout/train.csv")

y_data = train_file.values[:,1]
x_data = train_file.values[:,2:]

dpfold = int(len(y_data)/n_folds) # Number of data per fold
idx_data = np.arange(len(y_data))

# RMSE for each regression parameter lambda
i = 0
RMSE = np.zeros(len(lambdas))
for l in lambdas:
    RMSE_sum = 0.0
    # Cross validation for each fold
    for k in range(0,n_folds):

        # Use option 1 for picking training data, similar approach as in slides
        idx_test = idx_data[dpfold*k:dpfold*(k+1)] # Option 1, use 15 elements next to each other for one fold
        # idx_test = [idx+k for idx in idx_data if idx%n_folds == 0] # Option 2, use equally spaced data for one fold, here use every 10th element

        idx_train = [idx for idx in idx_data if idx not in idx_test]

        x_train = x_data[idx_train,:]
        y_train = y_data[idx_train]

        x_test = x_data[idx_test,:]
        y_test = y_data[idx_test]

        model = Ridge(alpha=l)
        model.fit(x_train,y_train)

        y_pred = model.predict(x_test)

        RMSE_sum += mean_squared_error(y_test, y_pred)**0.5

    RMSE[i] = RMSE_sum/n_folds # Average RMSE over all folds
    i = i + 1

np.savetxt("submission.csv",RMSE,comments='',delimiter=",",fmt="%s")