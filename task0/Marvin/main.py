'''
Introduction to Machine Learning
Task 0 (ungraded) : Linear regression
Team: Naiveoutliers
March 2021
'''

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# load data using pandas
training_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# =============
# training step
# =============
# extract data
Id_train = training_data['Id'].values
Y = np.matrix(training_data['y'].values)
Y = np.transpose(Y)
X = np.matrix(training_data[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']].values)

# compute LSQE regression estimate (DO NOT INVERT MATRIX!!!)
A = np.matmul(np.transpose(X),X)
b = np.matmul(np.transpose(X),Y)
w_predictor = np.linalg.solve(A,b)

# ===============
# prediction step
# ===============
Id_test = test_data['Id'].values
X_test = np.matrix(test_data[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']].values)
Y_test = np.matrix(np.matmul(X_test,w_predictor))

# ============
# compute RMSE 
# ============
Y_true = np.mean(X_test,1)
err = mean_squared_error(Y_test,Y_true)
print(err)

# ====================
# write result to file
# ====================
df = pd.DataFrame()
df['Id'] = Id_test
df['y'] = Y_test
df.to_csv('prediction.csv',sep=',',index=False)


