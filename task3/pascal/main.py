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
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001


# data loaders
class TrainData(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TestData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


# binary classification neural network
# TODO: improve
class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        self.layer_1 = nn.Linear(len(X_test_scaled[0]), 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


# train loop
def train(dataloader, model, criterion, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        epoch_acc = 0

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # compute loss and accuracy
            pred = model(X)
            loss = criterion(pred, y.unsqueeze(1))
            acc = f1_acc(y, pred)
            acc = binary_acc(y, pred)

            # TODO: Using f1 score build in function of sklearn gives an error, y_batch is binary and needs to be a
            #  float too, not resolved yet
            # vector = np.vectorize(np.float)
            # y_batch = vector(y_batch.unsqueeze(1).detach().numpy())
            # print(y_batch.unsqueeze(1).detach().numpy().astype(float))
            # acc = f1_score(y_batch.unsqueeze(1).to(torch.float32).detach().numpy(),y_pred.detach().numpy())

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            epoch_acc += acc

        print(f'Epoch {epoch + 0:03}: | '
              f'Loss: {epoch_loss / len(train_loader):.5f} | '
              f'Acc: {epoch_acc / len(train_loader):.3f}')


# model evaluation
def test(dataloader, model):
    y_pred = list()
    model.eval()
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            pred = model(X)
            pred = torch.sigmoid(pred)
            pred_tag = torch.round(pred)
            y_pred.append(pred_tag.cpu().numpy())
    return [a.squeeze().tolist() for a in y_pred]


# manual f1 score computation
def f1_acc(y_true, y_pred):
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1


# binary accuracy computation
def binary_acc(y_test, y_pred):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


# -----------------------------------------------------------------------------
# preprocessing
df_train = pd.read_csv("../handout/train.csv")
df_test = pd.read_csv("../handout/test.csv")

# split strings of amino acids, maintaining the sites
X_train = df_train['Sequence'].values
X_train = [list(X_train[i]) for i in range(len(X_train))]
X_test = df_test['Sequence'].values
X_test = [list(X_test[i]) for i in range(len(X_test))]

# get label of mutation
y_train = df_train['Active'].values

# encode data with one hot encoding, preserve the order of the mutation
enc = OneHotEncoder()
enc.fit(X_train)
X_train_encoded = enc.transform(X_train).toarray()
X_test_encoded = enc.transform(X_test).toarray()

# data is very imbalanced
# sns.countplot(x = 'Active', data=df_train)
# plt.show()

# scale data
scaler = StandardScaler()
# scaler = PowerTransformer()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# -----------------------------------------------------------------------------
# network architecture
train_data = TrainData(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train))
test_data = TestData(torch.FloatTensor(X_test_scaled))

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

# TODO: check this implementation, does not give better results, f1 seems off
# Weighted data loader
# counts = np.bincount(y_train)
# labels_weights = 1. / counts
# weights = labels_weights[y_train]
# train_sampler = WeightedRandomSampler(weights, len(weights))
# train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=False)

# set device for training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using {} device'.format(device))
model = BinaryClassification().to(device)
print(model)

# define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss() # TODO: check if correct
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------------------------------------------------------
# TODO: train and test model => try "train test split loop"
train(train_loader, model, criterion, optimizer)
predictions = test(test_loader, model)

# save predictions to csv file
np.savetxt("predictions.csv", predictions, fmt="%i")
