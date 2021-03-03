"""
Dummy Task 0
Team Naiveoutliers
Robin Schmid, Pascal Müller, Marvin Harms
Mar, 2021
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.set_printoptions(precision=3) # Printing values

# Load data
sample_file = pd.read_csv("../handout/sample.csv")
test_file = pd.read_csv("../handout/test.csv")
train_file = pd.read_csv("../handout/train.csv")

x_train = train_file.values[:, 2:]
y_train = train_file.values[:, 1]
x_test = test_file.values[:, 1:]

# Train model
model = LinearRegression()
model.fit(x_train,y_train)

# Predict output
y_pred = model.predict(x_test)

# Evaluate model
y_true = np.mean(x_test, 1)
RMSE = mean_squared_error(y_true, y_pred)**0.5

# Write to output file
submission = np.asarray([test_file.values[:,0].astype(int), y_pred]).transpose()
np.savetxt("submission.csv",submission,header="Id,y",comments='',delimiter=",",fmt="%i,%s")