"""
Task 3
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
May, 2021
"""

import numpy as np
import pandas as pd

# load data
df_train = pd.read_csv("../handout/train.csv")
df_test = pd.read_csv("../handout/test.csv")

# split strings of amino acids
X_train = df_train['Sequence'].values
X_train = [list(X_train[i]) for i in range(len(X_train))]
X_test = df_test['Sequence'].values
X_train = [list(X_test[i]) for i in range(len(X_test))]

# get label of mutation
y_train = df_train['Active'].values











