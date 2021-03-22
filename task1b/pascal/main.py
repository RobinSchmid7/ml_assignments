"""
Introduction to Machine Learning
Task 1b: Linear regression of feature transformations
Submission NLT021200BAPR21
Team Naiveoutliers
March 2021

Description of this file
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# load data
train_data = pd.read_csv("../handout/train.csv")

# split labels y and features x
y = train_data.values[:, 0]
X = train_data.values[:, 1:]

# store weights
weights = list()

# save weights to .csv file
np.savetxt('weights.csv', weights, fmt='%s')
