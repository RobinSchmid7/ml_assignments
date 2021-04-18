"""
Task 2
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
Apr, 2021
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.svm import SVR

df_train_features = pd.read_csv("../handout/train_features.csv")
df_train_labels = pd.read_csv("../handout/train_labels.csv")
df_test_features = pd.read_csv("../handout/test_features.csv")

# ================================
# load data into pandas dataframes
# ================================
df_train_features = df_train_features.set_index(["pid","Time"]).sort_index()
df_train_labels = df_train_labels.set_index(["pid"]).sort_index()
df_test_features = df_test_features.set_index(["pid","Time"]).sort_index()

available_threshold = 0.5


# get headers for Task 1
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

# get headers for Task 2
sepsis_label = ["LABEL_Sepsis"]
sepsis_features = [s[6:] for s in sepsis_label]

# get headers for Task 3
regression_labels = [
    "LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"
]
regression_features = [r[6:] for r in regression_labels]

# Idea for pre-processing:
# First, for each patient, flatten the data rows into one row, by taking the average of all non-nan measurements.
# Then, each dimension of the training data is normalized using standard scaler. For each missing measurement,
# we run a kNN classifier, copying the average of the k nearest neighbors to the missing value. 
# Closeness is evaluated based on frequent features as:
# Age, RRate, ABPm, SpO2, HR, ABPs

# ==========================
# Preprocessing training set
# ==========================
# eliminate time dependency
df_train_features = df_train_features.groupby('pid').mean()

# eliminate features, that are missing with more than 50% of the patients
num_features = len(df_train_features)
percentage_available = np.sum(~df_train_features.isnull()) / num_features
df_train_features = df_train_features[percentage_available[percentage_available > available_threshold].index.tolist()]

# TODO: remove later
print(df_train_features.columns)
print(df_train_features.shape)
print(df_train_labels.shape)

# apply standard scaler
feature_scaler = StandardScaler()
label_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(df_train_features.values)
Y_train = label_scaler.fit_transform(df_train_labels.values)

# replace missing values by KNN-imputation
imputer = KNNImputer(n_neighbors=3)
X_train = imputer.fit_transform(X_train)

# =========================
# Preprocessing of test set
# =========================
# TODO: add processing steps, identical to training set

# ============================
# Task 1: predict medical test
# ============================



