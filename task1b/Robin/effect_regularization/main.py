"""
Task 1b
Team Naiveoutliers
Robin Schmid, Pascal Müller, Marvin Harms
Mar, 2021

Use Lasso regularization and analyze effect on weights.
"""

import pandas as pd
import numpy as np
from sklearn import linear_model

# Regularization parameters
lasso_param = 0.001
ridge_param = 0.0000000001

full_data = pd.read_csv("../handout/train.csv")

y_data = full_data.values[:,1]
x_data = full_data.values[:,2:]

# Perform feature transformation
x_features = np.hstack((x_data, x_data**2, np.exp(x_data), np.cos(x_data), np.ones((x_data.shape[0],1))))

# Compare different models
lin_model = linear_model.LinearRegression(fit_intercept=False)
lin_model.fit(x_features, y_data)

las_model = linear_model.Lasso(alpha=lasso_param)
las_model.fit(x_features, y_data)

rid_model = linear_model.Ridge(alpha=ridge_param,fit_intercept=False)
rid_model.fit(x_features, y_data)

# Save weights
np.savetxt("linear.csv",lin_model.coef_,comments='',delimiter=",",fmt="%s")
np.savetxt("lasso.csv",las_model.coef_,comments='',delimiter=",",fmt="%s")
np.savetxt("rigde.csv",rid_model.coef_,comments='',delimiter=",",fmt="%s")