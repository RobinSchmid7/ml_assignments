"""
Task 3
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
May, 2021
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder,StandardScaler, PowerTransformer, LabelBinarizer, LabelEncoder


#######################
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 0.002
#######################

# set random seed
np.random.seed(42)
torch.manual_seed(42)

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

# Train data loader
class trainData(Dataset):
    '''data loader for training data'''
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

# Test data loader
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
        self.classifier1 = nn.Sequential(
            nn.Linear(len(X_test_enc[0]), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64,1)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(len(X_test_enc[0]), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32,1)
        )


    def forward(self, inputs):
        prediction = self.classifier2(inputs)
        return prediction

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

# ============
# load dataset
# ============
df_train = pd.read_csv("../handout/train.csv")
df_test = pd.read_csv("../handout/test.csv")
# get training data
X_train = df_train['Sequence'].values
X_test = df_test['Sequence'].values
# now, we split the strings into lists:
X_train = [list(X_train[i]) for i in range(len(X_train))]
X_test = [list(X_test[i]) for i in range(len(X_test))]

y_train = df_train['Active'].values

# ===================================================
# perform feature encoding of the dataset
# we choose a 4dim feature vector and ordinal encoder
# ===================================================
enc = OneHotEncoder(handle_unknown='error')
enc.fit(X_train)
X_train_enc = enc.transform(X_train).toarray()
X_test_enc = enc.transform(X_test).toarray()
print(enc.categories_)

#=================
# Standardize data
# ================
#scaler = StandardScaler()
scaler = PowerTransformer()
scaler.fit(X_train_enc)
X_train_enc = scaler.transform(X_train_enc)
X_test_enc = scaler.transform(X_test_enc)

# =========
# create NN
# =========
model = binaryClassification()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
# consider reweighting the classes due to heavily imbalanced dataset
pos_frac = np.sum(y_train)/len(y_train)
pos_weight = (1-pos_frac)/pos_frac
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# =================
# run training loop
# =================
model.train()
train_results = {}
test_results = {}
# iterate over all epochs
for epoch in range(1, EPOCHS+1):
    time_ = AverageMeter()
    loss_ = AverageMeter()
    acc_ = AverageMeter()
    # ===================
    # Define data loaders
    # ===================
    train_data = trainData(torch.FloatTensor(X_train_enc),
                        torch.FloatTensor(y_train))
    test_data = testData(torch.FloatTensor(X_test_enc))
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)
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
        acc = f1_acc(torch.round(torch.sigmoid(prediction)), labels.unsqueeze(1))
        loss_.update(loss.mean().item(), bs)
        acc_.update(acc.item(), bs)
        time_.update(time.time() - end)

    print(f'Epoch {epoch}. [Train] \t Time {time_.sum:.2f} Loss \
            {loss_.avg:.2f} \t Accuracy {acc_.avg:.2f}')
    train_results[epoch] = (loss_.avg, acc_.avg, time_.avg)

# plot training process
training = list()
for key, values in train_results.items():
    training.append([key, values[0], values[1], values[2]])
training = list(map(list, zip(*training)))

fig, axs = plt.subplots(2)
fig.suptitle('Loss and accuracy per epoch')
axs[0].plot(training[0], training[1], 'b')
axs[0].set_ylabel('loss')
axs[0].grid()
axs[1].plot(training[0], training[2], 'b')
axs[1].set_ylabel('accuracy')
axs[1].set_xlabel('epoch')
axs[1].grid()
plt.show()


# ===================
# run prediction loop
# ===================
y_pred_list = []
model.eval()
for data in test_loader:
    with torch.no_grad():
        features = data.to(device)
        prediction = model(features)
        prediction = torch.sigmoid(prediction)
        y_pred = torch.round(prediction)
        y_pred_list.append(y_pred.cpu().numpy())               

# =======================
# save predictions to csv
# =======================
y_pred_list = [batch.squeeze().tolist() for batch in y_pred_list]   
y_pred_list = np.concatenate(y_pred_list).ravel().tolist()           
np.savetxt("predictions.csv", y_pred_list, fmt="%i")
