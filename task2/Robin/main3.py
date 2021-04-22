"""
Task 2
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
Apr, 2021
"""

import pandas as pd
import numpy as np
import scipy
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm


############
# USER INPUT
############
use_preprocessed = False
available_threshold = 0.7
kernel = 'linear'

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
ROC_scores = []

if not use_preprocessed:
    df_train_features = pd.read_csv("../handout/train_features.csv")
    df_train_labels = pd.read_csv("../handout/train_labels.csv")
    df_test_features = pd.read_csv("../handout/test_features.csv")

    # ================================
    # load data into pandas dataframes
    # ================================
    print('loading raw data...')
    # df_train_features = df_train_features.set_index(["pid"]).sort_values(['pid', 'Time'])
    # df_test_features = df_test_features.set_index(["pid"]).sort_values(['pid', 'Time'])
    # df_train_labels = df_train_labels.set_index(["pid"]).sort_values(['pid'])

    # sort by pid and Time
    df_train_features = df_train_features.set_index(["pid", "Time"]).sort_index()
    df_train_labels = df_train_labels.set_index(["pid"]).sort_index()
    df_test_features = df_test_features.set_index(["pid", "Time"]).sort_index()
    print('...done')

    feature_scaler = StandardScaler()
    NNImputer = KNNImputer(n_neighbors=5)
    # MedianImputer = SimpleImputer(strategy='median')

    # ==========================
    # Preprocessing training set
    # ==========================
    print('processing training set...')
    # eliminate time dependency:
    # df_train_features.drop(labels='Time', axis=1, inplace=True)

    # eliminate features, that are missing with more than 90% of the patients
    num_features = len(df_train_features)
    percentage_available = np.sum(~df_train_features.isnull()) / num_features
    df_train_features = df_train_features[
        percentage_available[percentage_available > available_threshold].index.tolist()]

    # do not use age as feature with time dependency
    df_train_features = df_train_features.drop(columns="Age")

    # shift time such that it goes from 0 to 11
    df_train_features.index = pd.MultiIndex.from_arrays([df_train_features.index.get_level_values(0),
                                                         df_train_features.groupby(level=0).cumcount()],
                                                        names=["pid", "Time"])

    # unstack features with time dependency
    df_train_features = df_train_features.unstack(level=-1)

    # interpolate missing values
    print("interpolating training set...")
    for i in tqdm(range(len(df_train_features.columns))):
        df_train_features[df_train_features.columns[i:i + 12]] = df_train_features[
            df_train_features.columns[i:i + 12]].interpolate(axis=1, limit_direction='both', limit=12)
    print("interpolating finished...")

    # we average all values and append the gradient of the remaining columns:
    # RRate, Heartrate, ABPm, ABPd, SpO2
    # for column in ['RRate', 'Heartrate', 'ABPm', 'ABPd', 'SpO2']:
    #     # fill missing values
    #     df_train_features[column].groupby('pid', sort=False).fillna(method='ffill', inplace=True)
    #     df_train_features[column].groupby('pid', sort=False).fillna(method='bfill', inplace=True)
    #     # get the differences
    #     df_train_features[str('gradient_' + column)] = df_train_features[column].groupby('pid', sort=False).diff()
    #
    # # the average of the differences is the gradient
    # df_train_features = df_train_features.groupby('pid', sort=False).mean()

    # apply standard scaler
    X_train_scaled = feature_scaler.fit_transform(df_train_features.values)

    # replace remaining missing values with KNN
    X_train_scaled = NNImputer.fit_transform(X_train_scaled)
    df_train_features[df_train_features.columns] = X_train_scaled
    print('...done')

    # =========================
    # Preprocessing of test set
    # =========================
    print('processing test set...')
    # eliminate features, that are missing with more than 90% of the patients
    num_features = len(df_test_features)
    percentage_available = np.sum(~df_test_features.isnull()) / num_features
    df_test_features = df_test_features[
        percentage_available[percentage_available > available_threshold].index.tolist()]

    df_test_features = df_test_features.drop(columns="Age")

    df_test_features.index = pd.MultiIndex.from_arrays([df_test_features.index.get_level_values(0),
                                                         df_test_features.groupby(level=0).cumcount()],
                                                        names=["pid", "Time"])

    # Unstack features with few missing data to account for the time they were took
    df_test_features = df_test_features.unstack(level=-1)

    # Limit direction both parameter interpolates also if the value of the first column is missing!
    print("interpolating test set...")
    for i in tqdm(range(len(df_test_features.columns))):
        df_test_features[df_test_features.columns[i:i + 12]] = df_test_features[
            df_test_features.columns[i:i + 12]].interpolate(axis=1, limit_direction='both', limit=12)
    print("interpolating finished...")

    # apply standard scaler
    X_test_scaled = feature_scaler.transform(df_test_features.values)

    # replace missing values by KNN-imputation
    X_test_scaled = NNImputer.transform(X_test_scaled)
    df_test_features[df_test_features.columns] = X_test_scaled

    # # important: reorder index
    # df_ref = pd.read_csv("../handout/test_features.csv").set_index(["pid"])
    # df_test_features.reindex(df_ref.index.values.tolist())

    print('...done')

    # Intermittent step: save the preprocessed data
    df_train_features.to_csv('df_train_features.csv')
    df_train_labels.to_csv('df_train_labels.csv')
    df_test_features.to_csv('df_test_features.csv')
else:
    print('Loading preprocessed data...')
    df_train_features = pd.read_csv('df_train_features.csv')
    df_train_labels = pd.read_csv('df_train_labels.csv')
    df_test_features = pd.read_csv('df_test_features.csv')

    # df_train_features = df_train_features.set_index(["pid"]).sort_values(['pid'])
    # df_train_labels = df_train_labels.set_index(["pid"]).sort_values(['pid'])
    # df_test_features = df_test_features.set_index(["pid"])
    print('...done')

# =========================================
# split dataset to enable score evaluation:
# =========================================
# Dealing with imbalanced dataset
# sampler = RandomUnderSampler(random_state=42)
df_test_labels = pd.DataFrame(index=df_test_features.index)
df_test_labels.index.names = ['pid']

# ============================
# Task 1: predict medical test
# ============================
print('TASK1: predicting probabilities...')

# we let the svm fit and predict each test individually:
for label in classification_labels:
    # perform a randomized search CV
    y_train = df_train_labels[label].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_scaled,
        y_train,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    sampler = RandomUnderSampler(random_state=42)
    X_train, y_train = sampler.fit_resample(X_train, y_train)

    svm = SVC(probability=True, class_weight='balanced')

    # define parameter distributions
    param_dist = {
        'C': scipy.stats.expon(scale=1),
        'gamma': scipy.stats.expon(scale=1),
        'kernel': ['rbf']}

    # Perform randomized search cv to find optimal parameters
    svm_search = RandomizedSearchCV(
        svm,
        param_distributions=param_dist,
        cv=2,
        n_iter=8,
        scoring="roc_auc",
        error_score=0,
        verbose=3,
        n_jobs=-1)

    svm_search.fit(X_train, y_train)

    print(svm_search.best_estimator_.predict_proba(X_test)[:, 1])
    print(
        f"ROC score on test set "
        f"{roc_auc_score(y_test, svm_search.best_estimator_.predict_proba(X_test)[:, 1])}"
    )
    print(f"CV score {svm_search.best_score_}")
    print(f"Best parameters {svm_search.best_params_}")

    joblib.dump(
        svm_search.best_estimator_,
        f"svm_search_job_{classification_labels}.pkl",
    )
    ROC_scores.append(roc_auc_score(y_test, svm_search.best_estimator_.predict_proba(X_test)[:, 1]))
    y_pred = svm_search.best_estimator_.predict_proba(X_test_scaled)[:, 1]
    df_test_labels[label] = y_pred

print('...done')
print(ROC_scores)

# ======================
# Task 2: predict sepsis
# ======================
clf = RandomForestClassifier(n_estimators=1000, class_weight='balanced')
print('TASK2: predicting probabilities...')
y_train = df_train_labels[sepsis_label].astype(int).values
X_train, X_test, y_train, y_test = train_test_split(
    X_train_scaled,
    y_train,
    test_size=0.2,
    random_state=42,
    shuffle=True,
)
X_train, y_train = sampler.fit_resample(X_train, y_train)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
prob = clf.predict_proba(X_test)
# compute ROC curve and ROC area
print('compute ROC curve and ROC area...')
fpr, tpr, _ = roc_curve(y_test, prob[:, 1])
roc_auc = auc(fpr, tpr)
print(roc_auc)
# now, we perform the fit on the entire dataset:
clf.fit(df_train_features.to_numpy(), df_train_labels[sepsis_label].to_numpy())
print('ok')
y_pred = clf.predict_proba(X_test_scaled)[:, 1]
df_test_labels[sepsis_label] = y_pred
print('...done')

# ===========================
# Task 3: predict vital signs
# ===========================
svr = SVR(probability=True, class_weight='balanced')
print('TASK3: predicting regression values')

# define parameter distributions
param_dist = {
    'C': scipy.stats.expon(scale=1),
    'gamma': scipy.stats.expon(scale=1),
    'kernel': ['rbf']}

# Perform randomized search cv to find optimal parameters
svm_search = RandomizedSearchCV(
    svr,
    param_distributions=param_dist,
    cv=2,
    n_iter=10,
    scoring="r2",
    error_score=0,
    verbose=3,
    n_jobs=-1)

# we let the svr fit and predict each vital sign individually:
for label in regression_labels:
    print(label)
    # perform a randomized search CV
    y_train = df_train_labels[label].astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
        df_train_features,
        y_train,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    X_train, y_train = sampler.fit_resample(X_train, y_train)
    svm_search.fit(X_train, y_train)
    y_pred = svm_search.best_estimator_.predict(df_test_features)
    df_test_labels[label] = y_pred
print('...done')

# suppose df is a pandas dataframe containing the result
df_test_labels.to_csv('prediction.zip', index=True, float_format='%.3f', compression='zip')