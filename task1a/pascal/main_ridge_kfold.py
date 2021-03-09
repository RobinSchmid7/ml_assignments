import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge


# load data
train_data = pd.read_csv("../handout/train.csv")

# split labels y and features x
y = train_data.values[:, 0]
X = train_data.values[:, 1:]

# number of folds
n_folds = 10

# regularization parameters
param_lambda = [0.1, 1.0, 10.0, 100.0, 1000.0]

# store computed RMSE for each lambda
rmse = np.empty(len(param_lambda))

ii = 0
for param in param_lambda:
    # initialize RMSE to 0
    rmse[ii] = 0.0

    # perform cross validation
    kf = KFold(n_splits=n_folds, shuffle=False, random_state=None)
    for train_index, test_index in kf.split(X):
        # gather training data
        y_train = y[train_index]
        x_train = X[train_index]

        # gather test data
        y_true = y[test_index]
        x_test = X[test_index]

        # create ridge regression model
        model = Ridge(alpha=param, solver='cholesky')
        model.fit(x_train, y_train)

        # predict labels for test data
        y_pred = model.predict(x_test)

        # calculate RMSE
        rmse[ii] += mean_squared_error(y_true, y_pred)**0.5

    # average RMSE
    rmse[ii] /= n_folds
    print(rmse[ii])

    # increment
    ii += 1

# save computed RMSE to file
np.savetxt('submission_ridge_kfold.csv', rmse, fmt='%s')
