"""
Introduction to Machine Learning
Task 1a: Cross-validation for Ridge Regression
Submission NLT191200BMAR21
Team Naiveoutliers
March 2021

This approach uses sklearn Ridge regression and RepeatedKFold to
perform a 10-fold cross-validation of a given data set.  The model
is evaluated using the RMSE metric averaged over the 10 test folds.
The regression is performed on the original features, no feature
transformation and scaling are used.  The reported RMSE for all lambdas
are stored in a .csv file.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold


# load data
train_data = pd.read_csv("../handout/train.csv")

# split labels y and features x
y = train_data.values[:, 0]
X = train_data.values[:, 1:]

# number of folds
n_folds = 10
n_rep = 10000

# random state
rnd_state = 190221

# regularization parameters
param_lambda = [0.1, 1.0, 10.0, 100.0, 200.0]

# store computed RMSE for each lambda
rmse = np.empty(len(param_lambda))

ii = 0
for param in param_lambda:
    # initialize RMSE to 0
    rmse[ii] = 0.0

    # perform cross validation
    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_rep, random_state=rnd_state)
    for train_index, test_index in rkf.split(X):
        # gather training data
        y_train = y[train_index]
        x_train = X[train_index]

        # gather test data
        y_true = y[test_index]
        x_test = X[test_index]

        # create ridge regression model
        model = Ridge(alpha=param)
        model.fit(x_train, y_train)

        # predict labels for test data
        y_pred = model.predict(x_test)

        # calculate RMSE
        rmse[ii] += mean_squared_error(y_true, y_pred)**0.5

    # average RMSE
    rmse[ii] /= (n_folds * n_rep)
    print(rmse[ii])

    # increment
    ii += 1

# save computed RMSE to file
np.savetxt('ridge_rkf_rs'+str(rnd_state)+'_rep'+str(n_rep)+'.csv', rmse, fmt='%s')
