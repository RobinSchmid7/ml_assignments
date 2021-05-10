"""
Task 3
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
May, 2021
"""


# TODO:
# - f1 score for weighted random sampler seems wrong (bigger than 1)?
# - implement f1 score of sklearn, not own built in fct
# - f1 score seems very low now, not handling the imbalanced dataset well, maybe there are better ways than just using a
# scaler to the data?
# - try weighting in binary classification loss to deal with imbalanced classes - implemented, does not give better
# results, error?
# - try cross entropy loss instead of binary classification loss
# - power transformer instead of standard scaler for rescaling of the data?
# - later change model architecture, i.e. change layer sizes
# - try splitting of data s.t. every split has 1 and 0 (uniform)
# - try char to ASCII (reduce dimension to 4)


import time
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

# hyperparameters
EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 0.003

# set random seed
np.random.seed(42)
torch.manual_seed(42)


class AverageMeter:
    """ Computes and stores the average and current value. """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# training data loaders
class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


# test data loader
class TestData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


# binary classification neural network
class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        self.classifier1 = nn.Sequential(
            nn.Linear(len(X_test_scaled[0]), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(len(X_test_scaled[0]), 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1)
        )

    def forward(self, inputs):
        return self.classifier1(inputs)


# manual f1 score computation
def f1_acc(y_pred, y_true):
    tp = (y_true * y_pred).sum().float()
    tn = ((1 - y_true) * (1 - y_pred)).sum().float()
    fp = ((1 - y_true) * y_pred).sum().float()
    fn = (y_true * (1 - y_pred)).sum().float()
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1


# -----------------------------------------------------------------------------
# preprocessing
df_train = pd.read_csv("../handout/train.csv")
df_test = pd.read_csv("../handout/test.csv")

# data is very imbalanced
# sns.countplot(x = 'Active', data=df_train)
# plt.show()

# split strings of amino acids, maintaining the sites
X_train = df_train['Sequence'].values
X_train = [list(X_train[i]) for i in range(len(X_train))]
X_test = df_test['Sequence'].values
X_test = [list(X_test[i]) for i in range(len(X_test))]

# get activity label of mutations
y_train = df_train['Active'].values

# encode data with one hot encoding, preserve the order of the mutation
enc = OneHotEncoder()
enc.fit(X_train)
X_train_encoded = enc.transform(X_train).toarray()
X_test_encoded = enc.transform(X_test).toarray()
print(enc.categories_)

# scale data
scaler = StandardScaler()
# scaler = PowerTransformer()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# -----------------------------------------------------------------------------
# network architecture

# set device for training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using {} device'.format(device))
model = BinaryClassification().to(device)
print(model)

# define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# define loss function
# consider re-weighting the classes due to heavily imbalanced dataset
pos_frac = np.sum(y_train)/len(y_train)
pos_weight = (1-pos_frac)/pos_frac
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
# criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss() # TODO: check if correct
# criterion = nn.CrossEntropyLoss()

# data loaders
train_data = TrainData(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train))
test_data = TestData(torch.FloatTensor(X_test_scaled))
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

# weighted random sampler for training data => gives worse scoring!
# print('target train 0/1: {}/{}'.format(
#     len(np.where(y_train == 0)[0]), len(np.where(y_train == 1)[0])))
#
# class_sample_count = np.bincount(y_train)
#
# weight = 1. / class_sample_count
# samples_weight = np.array([weight[k] for k in y_train])
# samples_weight = torch.from_numpy(samples_weight)
# samples_weight = samples_weight.double()
#
# sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
#
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler)
#
# for i, (data, target) in enumerate(train_loader):
#     print("batch index {}, 0/1: {}/{}".format(
#         i,
#         len(np.where(target.numpy() == 0)[0]),
#         len(np.where(target.numpy() == 1)[0])))

# -----------------------------------------------------------------------------
# training loop
model.train()
train_results = {}

# iterate over all epochs
for epoch in range(1, EPOCHS+1):
    time_ = AverageMeter()
    loss_ = AverageMeter()
    acc_ = AverageMeter()

    # iterate over training mini-batches
    for i, data, in enumerate(train_loader, 1):
        # accounting
        end = time.time()
        features, labels = data
        features = features.to(device)
        labels = labels.to(device)
        bs = features.size(0)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward propagation
        prediction = model(features)
        # compute the loss
        loss = criterion(prediction, labels.unsqueeze(1))
        # backward propagation
        loss.backward()
        # optimization step
        optimizer.step()

        # accounting
        acc = f1_acc(torch.round(torch.sigmoid(prediction)), labels.unsqueeze(1))
        loss_.update(loss.mean().item(), bs)
        acc_.update(acc.item(), bs)
        time_.update(time.time() - end)

    print(f'Epoch {epoch}. [Train] \t Time {time_.sum:.2f} Loss \
            {loss_.avg:.2f} \t Accuracy {acc_.avg:.2f}')

    train_results[epoch] = (loss_.avg, acc_.avg, time_.avg)

# -----------------------------------------------------------------------------
# perform predictions
model.eval()
y_pred_list = list()
for data in test_loader:
    with torch.no_grad():
        features = data.to(device)
        prediction = model(features)
        y_pred = torch.round(torch.sigmoid(prediction))
        y_pred_list.append(y_pred.cpu().numpy())

# save predictions to csv file
y_pred_list = [batch.squeeze().tolist() for batch in y_pred_list]
np.savetxt("predictions.csv", y_pred_list, fmt="%i")
