"""
Task 2
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
Apr, 2021
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC
from sklearn.svm import NuSVR

df_train_features = pd.read_csv("../handout/train_features.csv")
df_train_labels = pd.read_csv("../handout/train_labels.csv")
df_test_features = pd.read_csv("../handout/test_features.csv")

# Use pid and Time as index and sort
df_train_features = df_train_features.set_index(["pid","Time"]).sort_index()
df_train_labels = df_train_labels.set_index(["pid"]).sort_index()
df_test_features = df_test_features.set_index(["pid","Time"]).sort_index()

available_threshold = 0.75

# Task 1
classification_labels = [
    "LABEL_BaseExcess",
    "LABEL_Fibrinogen",
    "LABEL_AST",
    "LABEL_Alkalinephos",
    "LABEL_Bilirubin_total",
    "LABEL_Lactate",
    "LABEL_TroponinI",
    "LABEL_SaO2",
    "LABEL_Bilirubin_direct",
    "LABEL_EtCO2"
]
classification_features = [c[6:] for c in classification_labels]

# Task 2
sepsis_label = ["Label_Sepsis"]
sepsis_features = [s[6:] for s in sepsis_label]

# Task 3
regression_labels = [
    "LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"
]
regression_features = [r[6:] for r in regression_labels]

# Idea for pre-processing:
# For features for which the data is available for more than a certain percentage use each time series separately. Interplate to get the missing values.
# In first step do not use other features, later try to use features with a lot of missing data and use the aveage over all time steps for these features.
# Try to explore if using a mask helps - a binary signal which tell if the given feature has a nan value or not

### Preprocessing training set

# Only use features of data if the data for this feature is available more than a certain threshold
num_data = len(df_train_features)

# Percentage of data available of each feature
percentage_available = np.sum(~df_train_features.isnull()) / num_data

# Reduced feature set
df_train_features = df_train_features[percentage_available[percentage_available > available_threshold].index.tolist()]
# Do not use Age as a feature where time matters //TODO: remove more elegantly
df_train_features = df_train_features.drop(columns="Age")
# Remaining features: Age, RRate, ABPm, SpO2, Heartrate, ABPs

# Shift data such that the time series goes from 0 to 11
df_train_features.index = pd.MultiIndex.from_arrays([df_train_features.index.get_level_values(0),
                                                     df_train_features.groupby(level=0).cumcount()],
                                                    names=["pid","Time"])
# Unstack features with few missing data to account for the time they were took
df_train_features = df_train_features.unstack(level=-1)

# Interpolate missing values
#TODO: check missing values in pre processed data table - some are not interpolated? what is the problem for this??
#TODO: speed this up (openmp for python?)

# print("Start interpolating...")
# for i in range(len(df_train_features.columns)):
#     df_train_features[df_train_features.columns[i:i+12]] = df_train_features[df_train_features.columns[i:i+12]].interpolate(axis=1)
# print("Interpolating finished")
#
# # Load saved interpolated data for quicker debugging
# TODO: remove this in the end

# np.savetxt("filtered_train_features.csv",df_train_features,comments='',delimiter=",",fmt="%s")

# For debugging load data since interpolating takes some time
df_train_features = pd.read_csv("filtered_train_features.csv")

### Preprocessing test set

num_data = len(df_test_features)

# Percentage of data available of each feature
percentage_available = np.sum(~df_test_features.isnull()) / num_data

# Reduced feature set
df_test_features = df_test_features[percentage_available[percentage_available > available_threshold].index.tolist()]
# Do not use Age as a feature where time matters //TODO: remove more elegantly
df_test_features = df_test_features.drop(columns="Age")
# Remaining features: Age, RRate, ABPm, SpO2, Heartrate, ABPs

# Shift data such that the time series goes from 0 to 11
df_test_features.index = pd.MultiIndex.from_arrays([df_test_features.index.get_level_values(0),
                                                     df_test_features.groupby(level=0).cumcount()],
                                                    names=["pid","Time"])
# Unstack features with few missing data to account for the time they were took
df_test_features = df_test_features.unstack(level=-1)

# Interpolate missing values
#TODO: check missing values in pre processed data table - some are not interpolated? what is the problem for this??
#TODO: speed this up (openmp for python?)

# print("Start interpolating...")
# for i in range(len(df_test_features.columns)):
#     df_test_features[df_test_features.columns[i:i+12]] = df_test_features[df_test_features.columns[i:i+12]].interpolate(axis=1)
# print("Interpolating finished")

# Load saved interpolated data for quicker debugging
# TODO: remove this in the end

# np.savetxt("filtered_test_features.csv",df_test_features,comments='',delimiter=",",fmt="%s")

# For debugging load data since interpolating takes some time
df_test_features = pd.read_csv("filtered_train_features.csv")

# Scale data, equalizes one sided distribution
scaler = StandardScaler()
scaler.transform(df_train_features)

# Try using SVM for classification
clf = NuSVC(kernel='linear')
clf.fit(df_train_features,df_train_labels)

clf.predict(df_test_features)

pass