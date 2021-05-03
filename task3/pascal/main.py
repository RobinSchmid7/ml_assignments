"""
Task 3
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
May, 2021
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score

# load data
df_train = pd.read_csv("../handout/train.csv")
df_test = pd.read_csv("../handout/test.csv")

# split strings of amino acids, maintaining the sites
X_train = df_train['Sequence'].values
X_train = [list(X_train[i]) for i in range(len(X_train))]
X_test = df_test['Sequence'].values
X_test = [list(X_test[i]) for i in range(len(X_test))]

# get label of mutation
y_train = df_train['Active'].values

# use one hot encoder for the mutations
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train)
X_train_encoded = enc.transform(X_train).toarray()
