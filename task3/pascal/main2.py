"""
Task 3
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
May, 2021
"""


# TODO:
# - implement f1 score of sklearn, not own built in fct
# - try cross entropy loss instead of binary classification loss
# - later change model architecture, i.e. change layer sizes
# - try char to ASCII (reduce dimension to 4)

# DONE/TESTED:
# - f1 score for weighted random sampler seems wrong (bigger than 1)? => tensor to numpy
# - f1 score seems very low now, not handling the imbalanced dataset well, maybe there are better ways than just using a
#   scaler to the data? => power transformer instead of standard scaler
# - try weighting in binary classification loss to deal with imbalanced classes - implemented, does not give better
#   results, error? => seems like that distribution is changed very extreme
# - try splitting of data s.t. every split has 1 and 0 (uniform) => seems like that distribution is changed very extreme
# - power transformer instead of standard scaler for rescaling of the data? => slightly better results

import time as timing
import seaborn as sns

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# hyperparameters
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TEST_SIZE = 0.2

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
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


# evaluation data loader
class EvalData(Dataset):
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
            nn.Linear(len(X_test[0]), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(len(X_test[0]), 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1)
        )

    def forward(self, inputs):
        return self.classifier1(inputs)


# performs training of the model
def train(trainloader, device, model, optimizer):
    time_ = AverageMeter()
    loss_ = AverageMeter()
    acc_ = AverageMeter()

    model.train()

    for i, data, in enumerate(trainloader, 1):
        # accounting
        end = timing.time()

        # get the features and labels
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
        time_.update(timing.time() - end)

    return loss_, acc_, time_


# performs testing of the model
def test(testloader, device, model):
    time_ = AverageMeter()
    loss_ = AverageMeter()
    acc_ = AverageMeter()

    model.eval()

    for i, data, in enumerate(testloader, 1):
        # accounting
        end = timing.time()

        # get the features and labels
        features, labels = data
        features = features.to(device)
        labels = labels.to(device)
        bs = features.size(0)

        with torch.no_grad():
            prediction = model(features)
            loss = criterion(prediction, labels.unsqueeze(1))
            acc = f1_acc(torch.round(torch.sigmoid(prediction)), labels.unsqueeze(1))

            print(classification_report(torch.round(torch.sigmoid(prediction)), labels.unsqueeze(1)))
            print(confusion_matrix(torch.round(torch.sigmoid(prediction)), labels.unsqueeze(1)))

            # accounting
            loss_.update(loss.mean().item(), bs)
            acc_.update(acc.mean().item(), bs)
            time_.update(timing.time() - end)

    return loss_, acc_, time_


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
df_data = pd.read_csv("../handout/train.csv")
df_eval = pd.read_csv("../handout/test.csv")

# data is very imbalanced
# sns.countplot(x = 'Active', data=df_train)
# plt.show()

# split strings of amino acids, maintaining the sites
X_data = df_data['Sequence'].values
X_data = [list(X_data[i]) for i in range(len(X_data))]
X_eval = df_eval['Sequence'].values
X_eval = [list(X_eval[i]) for i in range(len(X_eval))]

# get activity label of mutations
y_data = df_data['Active'].values

# encode data with one hot encoding, preserve the order of the mutation
enc = OneHotEncoder()
enc.fit(X_data)
X_data = enc.transform(X_data).toarray()
X_eval = enc.transform(X_eval).toarray()
print(enc.categories_)

# scale data
# scaler = StandardScaler()
scaler = PowerTransformer()
X_data = scaler.fit_transform(X_data)
X_eval = scaler.transform(X_eval)

# perform train test split on data
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=TEST_SIZE, random_state=42)

# -----------------------------------------------------------------------------
# network architecture

# set device for training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using {} device'.format(device))
model = BinaryClassification().to(device)
print(model)

# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# define loss function
# consider re-weighting the classes due to heavily imbalanced dataset
pos_frac = np.sum(y_train)/len(y_train)
pos_weight = (1-pos_frac)/pos_frac
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

# data loaders
train_data = TrainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
test_data = TestData(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
eval_data = EvalData(torch.FloatTensor(X_eval))
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(dataset=eval_data, batch_size=1)

# -----------------------------------------------------------------------------
# training the model
train_results = {}
test_results = {}

# initial test error
# loss, acc, time = test(test_loader, device, model)
# print(f'Upon initialization. [Test] \t Time {time.avg:.2f} Loss {loss.avg:.2f} \t Accuracy {acc.avg:.2f}')
# test_results[0] = (loss.avg, acc.avg, time.avg)

# iterate overall epochs
for epoch in range(1, EPOCHS + 1):
    # training the model
    loss, acc, time = train(train_loader, device, model, optimizer)
    print(f'Epoch {epoch}. [Train] \t Time {time.sum:.2f} Loss {loss.avg:.2f} \t Accuracy {acc.avg:.2f}')
    train_results[epoch] = (loss.avg, acc.avg, time.avg)

    # testing the model
    if not (epoch % 2):
        loss, acc, time = test(test_loader, device, model)
        print(f'Epoch {epoch}. [Test] \t Time {time.sum:.2f} Loss {loss.avg:.2f} \t Accuracy {acc.avg:.2f}')
        test_results[epoch] = (loss.avg, acc.avg, time.avg)

# plot the training process
training = list()
for key, values in train_results.items():
    training.append([key, values[0], values[1], values[2]])
training = list(map(list, zip(*training)))

testing = list()
for key, values in test_results.items():
    testing.append([key, values[0], values[1], values[2]])
testing = list(map(list, zip(*testing)))

fig, axs = plt.subplots(2)
fig.suptitle('Loss and accuracy per epoch')
axs[0].plot(training[0], training[1], 'b',
            testing[0], testing[1], 'ro')
axs[0].set_ylabel('loss')
axs[0].grid()
axs[1].plot(training[0], training[2], 'b',
            testing[0], testing[2], 'ro')
axs[1].set_ylabel('accuracy')
axs[1].set_xlabel('epoch')
axs[1].grid()
plt.show()

# -----------------------------------------------------------------------------
# perform predictions on evaluation set
model.eval()
y_pred_list = list()
for data in eval_loader:
    with torch.no_grad():
        features = data.to(device)
        prediction = model(features)
        y_pred = torch.round(torch.sigmoid(prediction))
        y_pred_list.append(y_pred.cpu().numpy())

# save predictions to csv file
y_pred_list = [batch.squeeze().tolist() for batch in y_pred_list]
np.savetxt("predictions.csv", y_pred_list, fmt="%i")
