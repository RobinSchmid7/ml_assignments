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
from sklearn.svm import LinearSVC
from sklearn.svm import SVR

def softmax(x):
    '''return the softmax of a vector x'''
    return 1/(1+np.exp(x))

############
# USER INPUT
############
use_preprocessed = False
available_threshold = 0.3
kernel= 'linear'

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
sepsis_label = "LABEL_Sepsis"
sepsis_features = [s[6:] for s in sepsis_label]

# get headers for Task 3
regression_labels = [
"LABEL_RRate", "LABEL_ABPm", "LABEL_SpO2", "LABEL_Heartrate"
]
regression_features = [r[6:] for r in regression_labels]

if not use_preprocessed:
    df_train_features = pd.read_csv("../handout/train_features.csv")
    df_train_labels = pd.read_csv("../handout/train_labels.csv")
    df_test_features = pd.read_csv("../handout/test_features.csv")

    # ================================
    # load data into pandas dataframes
    # ================================
    df_train_features = df_train_features.set_index(["pid"]).sort_index()
    df_train_labels = df_train_labels.set_index(["pid"]).sort_index()
    df_test_features = df_test_features.set_index(["pid"])

    # Idea for pre-processing:
    # First, for each patient, flatten the data rows into one row, by taking the average of all non-nan measurements.
    # Then, we run a kNN classifier, copying the average of the k nearest neighbors to the missing value. 
    # Closeness is evaluated based on all other, non-nan features

    # ==========================
    # Preprocessing training set
    # ==========================
    # eliminate time dependency: 
    df_train_features.drop('Time',axis=1)
    # we average all values and append the gradient of the following columns:
    # RRate, Heartrate, ABPm, ABPd, SpO2
    for column in ['RRate','Heartrate','ABPm','ABPd','SpO2']:
        df_train_features[str('gradient_'+column)] = df_train_features[column].groupby('pid',sort=False).diff()

    df_train_features = df_train_features.groupby('pid',sort=False).mean()

    # eliminate features, that are missing with more than 50% of the patients
    num_features = len(df_train_features)
    percentage_available = np.sum(~df_train_features.isnull()) / num_features
    df_train_features = df_train_features[percentage_available[percentage_available > available_threshold].index.tolist()]

    # apply standard scaler
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(df_train_features.values)

    # replace missing values by KNN-imputation
    imputer = KNNImputer(n_neighbors=3)
    X_train = imputer.fit_transform(X_train)
    df_train_features[df_train_features.columns] = X_train


    # =========================
    # Preprocessing of test set
    # =========================
    # eliminate time dependency
    df_test_features.drop('Time',axis=1)
    # we average all values and append the gradient of the following columns:
    # RRate, Heartrate, ABPm, ABPd, SpO2
    for column in ['RRate','Heartrate','ABPm','ABPd','SpO2']:
        df_test_features[str('gradient_'+column)] = df_test_features[column].groupby('pid',sort=False).diff()
    df_test_features = df_test_features.groupby('pid',sort=False).mean()

    # eliminate features, that are missing with more than 50% of the patients
    df_test_features = df_test_features[percentage_available[percentage_available > available_threshold].index.tolist()]

    # apply standard scaler
    X_test = feature_scaler.transform(df_test_features.values)

    # replace missing values by KNN-imputation
    X_test = imputer.transform(X_test)
    df_test_features[df_test_features.columns] = X_test

    # Intermittent step: save the preprocessed data
    df_train_features.to_csv('df_train_features.csv')
    df_train_labels.to_csv('df_train_labels.csv')
    df_test_features.to_csv('df_test_features.csv')

else:
    df_train_features = pd.read_csv('df_train_features.csv')
    df_train_labels = pd.read_csv('df_train_labels.csv')
    df_test_features = pd.read_csv('df_test_features.csv')
    df_train_features = df_train_features.set_index(["pid"]).sort_index()
    df_train_labels = df_train_labels.set_index(["pid"]).sort_index()
    df_test_features = df_test_features.set_index(["pid"])


# ============================
# Task 1: predict medical test
# ============================
df_test_labels = pd.DataFrame(index=df_test_features.index)
df_test_labels.index.names = ['pid']
svm = SVC(probability=True,class_weight='balanced')
print('TASK1: predicting probabilities...')
# we let the linear svm fit and predict each test individually:
for label in classification_labels:
    svm.fit(df_train_features.to_numpy(),df_train_labels[label].to_numpy())
    # compute probabilities
    pred = svm.predict(df_test_features.to_numpy())
    prob = svm.predict_proba(df_test_features.to_numpy())
    df_test_labels[label] = [p[1] for p in prob]
print('...done')


# ======================
# Task 2: predict sepsis
# ======================
svm = SVC(probability=True,class_weight='balanced')
print('TASK2: predicting probabilities...')
svm.fit(df_train_features.to_numpy(),df_train_labels[sepsis_label].to_numpy())
pred = svm.predict(df_test_features.to_numpy())
prob = svm.predict_proba(df_test_features.to_numpy())
df_test_labels[sepsis_label] = [p[1] for p in prob]
print('...done')

# ===========================
# Task 3: predict vital signs
# ===========================
svm = SVR(kernel='rbf')
print('TASK3: predicting regression values')
# we let the svr fit and predict each vital sign individually:
for label in regression_labels:
    svm.fit(df_train_features.to_numpy(),df_train_labels[label].to_numpy())
    # compute distance to hyperplane
    pred = svm.predict(df_test_features.to_numpy())
    df_test_labels[label] = pred
print('...done')

# suppose df is a pandas dataframe containing the result
df_test_labels.to_csv('prediction.zip', index=True, float_format='%.3f',compression='zip')
# try to evaluate the performance:
# we do 5-fold cross-validation:

# TODO: include 5-fold CV test



