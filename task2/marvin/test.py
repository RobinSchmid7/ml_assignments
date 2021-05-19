"""
Task 2
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
Apr, 2021
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

TESTS = [
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

VITALS = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']

SEPSIS = ['LABEL_Sepsis']

# user input
label = SEPSIS[0]


# load preprocessed data
print('load data...')
df_train_features = pd.read_csv('df_train_features.csv').set_index('pid').sort_index()
df_train_labels = pd.read_csv('df_train_labels.csv').set_index('pid').sort_index()
print('...done')

# split data
print('split data...')
X_train, X_test, y_train, y_test = train_test_split(df_train_features.to_numpy(), df_train_labels[label].to_numpy(),
                                                    test_size=0.5)
print('...done')

# predict test for label
print('predicting probabilities...')
svm = RandomForestClassifier(n_estimators=1000,class_weight='balanced')
svm.fit(X_train, y_train)
pred = svm.predict(X_test)
prob = svm.predict_proba(X_test)
print('...done')

# compute ROC curve and ROC area
print('compute ROC curve and ROC area...')
fpr, tpr, _ = roc_curve(y_test, prob[:, 1])
roc_auc = auc(fpr, tpr)
print('...done')

# plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC ' + label)
plt.legend(loc='lower right')
plt.savefig('plots/'+label+'.png')
plt.show()
