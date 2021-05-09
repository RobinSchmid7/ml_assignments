"""
Task 3
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
May, 2021
"""
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder,StandardScaler


class AverageMeter():
    """Computes and stores the average and current value"""
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

# Data loaders
# Train data
class trainData(Dataset):
    '''data loader for training data'''
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


# Test data
class testData(Dataset):
    '''data loader for test data'''
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)

# Architecture
class binaryClassification(nn.Module):
    '''class to define NN architecture'''
    def __init__(self):
        super(binaryClassification, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(len(X_test_enc[0]), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self, inputs):
        prediction = self.classifier(inputs)
        return prediction

def train(train_loader, device, model, optimizer):
    time_ = AverageMeter()
    loss_ = AverageMeter()
    acc_ = AverageMeter()
    model.train()
    # iterate over training minibatches
    for i, data, in enumerate(train_loader, 1):
        # Accounting
        end = time.time()
        features,labels = data
        features = features.to(device)
        labels = labels.to(device)
        bs = features.size(0)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward propagation
        prediction = model(features) 
        # compute the loss
        loss = criterion(prediction, labels.unsqueeze(1))
        # Backward propagation
        loss.backward()
        # Optimization step. 
        optimizer.step()

        # Accounting
        acc = f1_acc(torch.round(prediction), labels.unsqueeze(1))
        loss_.update(loss.mean().item(), bs)
        acc_.update(acc.item(), bs)
        time_.update(time.time() - end)

    return loss_, acc_, time_

def test(device, model):
    time_ = AverageMeter()
    loss_ = AverageMeter()
    acc_ = AverageMeter()
    model.eval()

    for i, data, in enumerate(test_loader, 1):
        # Accounting
        end = time.time()
        features, labels = data
        features = features.to(device)
        labels = labels.to(device)

        bs = features.size(0)

        with torch.no_grad():
            prediction = model(features)
            loss = criterion(prediction, labels.unsqueeze(1))
            acc = f1_acc(torch.round(prediction), labels.unsqueeze(1))
    
            # Accounting
            loss_.update(loss.mean().item(), bs)
            acc_.update(acc.mean().item(), bs)
            time_.update(time.time() - end)

    return loss_, acc_, time_

def experiment(num_epochs, train_loader, device, model, optimizer):
    train_results = {}
    test_results = {}
    # Initial test error
    loss, acc, time = test(device, model)
    print(f'Upon initialization. [Test] \t Time {time.avg:.2f} \
            Loss {loss.avg:.2f} \t Accuracy {acc.avg:.2f}')
    test_results[0] = (loss, acc, time)
    

    for epoch in range(1, num_epochs+1):
        loss, acc, time = train(train_loader, device, model, optimizer)
        print(f'Epoch {epoch}. [Train] \t Time {time.sum:.2f} Loss \
                {loss.avg:.2f} \t Accuracy {acc.avg:.2f}')
        train_results[epoch] = (loss.avg, acc.avg, time.avg)

        if not (epoch % 2):
          loss, acc, time = test(device, model)
          print(f'Epoch {epoch}. [Test] \t Time {time.sum:.2f} Loss \
                {loss.avg:.2f} \t Accuracy {acc.avg:.2f}')
          test_results[epoch] = (loss.avg, acc.avg, time.avg)

    return train_results, test_results

# Define accuracy function
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

# define F1 score function
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

def get_weight(labels):
    global pos_weight
    global neg_weight
    weights = []
    for label in labels:
        if label:
            weights.append(pos_weight)
        else:
            weights.append(neg_weight)
    return torch.tensor(weights)
#######################
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
#######################

# ============
# load dataset
# ============
df_train = pd.read_csv("../handout/train.csv")
df_test = pd.read_csv("../handout/test.csv")
# get training data
X_train = df_train['Sequence'].values
y_train = df_train['Active'].values
#get test data
X_test = df_test['Sequence'].values
# now, we split the strings into lists:
X_train = [list(X_train[i]) for i in range(len(X_train))]
X_test = [list(X_test[i]) for i in range(len(X_test))]

# ===================================================
# perform feature encoding of the dataset
# we choose a 4dim feature vector and ordinal encoder
# ===================================================
enc = OrdinalEncoder(handle_unknown='error')
enc.fit(X_train)
X_train_enc = enc.transform(X_train)
X_test_enc = enc.transform(X_test)
print(enc.categories_)

# =========
# create NN
# =========
model = binaryClassification()
device='cpu'
model.to(device)
# consider reweighting the classes due to heavily imbalanced dataset
pos_frac = np.sum(y_train)/len(y_train)
pos_weight = 1/pos_frac
neg_weight = 1 - pos_weight
class_weights = torch.tensor([neg_weight,pos_weight])
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
optimizer = optim.ASGD(model.parameters(), lr=LEARNING_RATE)

#=================
# Standardize data
# ================
scaler = StandardScaler()
X_train_enc = scaler.fit_transform(X_train_enc)
X_test_enc = scaler.transform(X_test_enc)

# ===========================
# run 4-fold cross-validation
# ===========================
kf = KFold(n_splits=5, shuffle=True, random_state=123)
for (trainset,testset) in kf.split(X_train_enc,y=y_train):


    # ===================
    # Define data loaders
    # ===================
    train_data = trainData(torch.FloatTensor(X_train_enc[trainset]),
                        torch.FloatTensor(y_train[trainset]))
    test_data = trainData(torch.FloatTensor(X_train_enc[testset]),
                        torch.FloatTensor(y_train[testset]))
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)


    # =================
    # run training loop
    # =================
    train_res_0, test_res_0 = experiment(num_epochs=EPOCHS, 
                                    train_loader=train_loader, 
                                    device=device, 
                                    model=model, 
                                    optimizer=optimizer)

# ==========================
# after training, we predict
# ==========================
test_data = testData(torch.FloatTensor(X_test_enc))
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
y_pred_list = []
model.eval()
for data in test_loader:
    with torch.no_grad():
        features = data
        features = features.to(device)
        bs = features.size(0)
        prediction = model(features)
        prediction = nn.sigmoid(prediction)
        y_pred = torch.round(prediction)
        y_pred_list.append(y_pred.cpu().numpy().T)               

# =======================
# save predictions to csv
# =======================
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]                             
np.savetxt("predictions.csv", y_pred_list, fmt="%i")