"""
Task 1b
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
Mar, 2021
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
import matplotlib.pyplot as plt

analyse_plots = True

# Regularization parameters
elnet_lambdas = [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999]
las_lambdas = np.linspace(0.001,0.01,100)
rid_lambdas = np.linspace(0.01,50,100)

# Load data
full_data = pd.read_csv("../handout/train.csv")
y_data = full_data.values[:, 1]
x_data = full_data.values[:, 2:]

# Use feature transformations
x_transformed = np.hstack((x_data, np.square(x_data), np.exp(x_data), np.cos(x_data), np.ones((x_data.shape[0],1))))

# Perform feature transformation, use CV for evaluation
# Compare different models
lin_model = LinearRegression(fit_intercept=False)
lin_model.fit(x_transformed, y_data)

elnet_model = ElasticNetCV(l1_ratio=elnet_lambdas,n_alphas=100,fit_intercept=False)
elnet_model.fit(x_transformed,y_data)

las_model = LassoCV(alphas=las_lambdas,cv=2,fit_intercept=False) # Default: 5 folds
las_model.fit(x_transformed, y_data)
las_mean_error = np.mean(las_model.mse_path_,1)

rid_model = RidgeCV(alphas=rid_lambdas,fit_intercept=False,store_cv_values=True)
rid_model.fit(x_transformed, y_data)
rid_mean_error = np.mean(rid_model.cv_values_,0)

# Save weights, use Ridge CV for submission
np.savetxt("linear.csv",lin_model.coef_,comments='',delimiter=",",fmt="%s")
np.savetxt("elnet.csv",elnet_model.coef_,comments='',delimiter=",",fmt="%s")
np.savetxt("lasso.csv",las_model.coef_,comments='',delimiter=",",fmt="%s")
np.savetxt("rigde.csv",rid_model.coef_,comments='',delimiter=",",fmt="%s")

# Analyse evolution of RMSE for different alpha for Lasso and Ridge CV
if analyse_plots:
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.plot(las_lambdas,las_mean_error)
    ax1.set_title("Lasso CV")
    ax1.set(xlabel="lambda",ylabel="RMSE")
    ax2.plot(rid_lambdas, rid_mean_error)
    ax2.set_title("Ridge CV")
    ax2.set(xlabel="lambda", ylabel="RMSE")
    plt.show()