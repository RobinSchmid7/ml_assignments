"""
Task 4: Image taste classification
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
May, 2021
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import os,time
import matplotlib.pyplot as plt
# example of using a pre-trained model as a classifier
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from keras.applications.resnet50 import ResNet50
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split



#######################
USE_PREPROCESSED = True
LEARNING_RATE = 0.0002
BATCHSIZE = 64
EPOCHS = 10
#######################

# set random seed
np.random.seed(42)


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
        self.classifier = nn.Sequential(
            nn.Linear(len(X_train[0]), 256),
            nn.ReLU(),
            #nn.Linear(256, 256),
            #nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64,1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        prediction = self.classifier2(inputs)
        return prediction


def get_image(filename):
    global handout_path
    img = load_img(handout_path+'/food/'+filename, target_size=(224, 224))
    img = img_to_array(img)
    return img

#=======================================================================================
# IDEA: PREDICT CLASS LABELS FOR EACH IMAGE INDEPENDENTLY.
# THEN, WE SAVE THE 1000-DIM OUTPUT VECTOR OF EACH IMAGE AS A NEW FEATURE VECTOR.
# WE THEN TRAIN A NEURAL NETWORK WITH 3x1000 INPUTS AND ONE SINGLE OUTPUT 
# TO PREDICT A TASTE SIMILARITY BETWEEN THE TWO INPUTS IN RANGE [0,1].
#=======================================================================================

handout_path = "/home/marvin/Downloads/IML_handout_task4/"


if not USE_PREPROCESSED:
    # ===========
    # setup model
    # ===========
    # we use the pretrained model
    model = tf.keras.Sequential()
    model = ResNet50(include_top=True)
    model.summary()

    # =====================================
    # predict image classes (preprocessing)
    # =====================================
    header = []
    for i in range(1000):
        header.append('class'+str(i+1))

    df = pd.DataFrame(columns=header)
    imagenames = sorted(os.listdir(handout_path+'food/'))
    for filename in sorted(os.listdir(handout_path+'food/')):
        if filename.endswith('.jpg'):
            print('\n'+filename)
            image = get_image(filename)
            # reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # prepare the image for the model
            image = preprocess_input(image)
            # predict the probability across all output classes
            yhat = model.predict(image)
            # retrieve the most likely result, e.g. highest probability
            label = decode_predictions(yhat)
            # retrieve the most likely result, e.g. highest probability
            label = label[0][0]
    
            print('%s (%.2f%%)' % (label[1], label[2]*100))
            # append the probabilities to dataframe
            df.loc[filename[0:-4]] = yhat[0][:]

    df.to_csv(handout_path+'class_probabilities.csv')

else:
    # ========================
    # load preprocessed images
    # ========================
    df = pd.read_csv(handout_path+'class_probabilities.csv',index_col=0)

# =====================================
# split the data in train and test sets
# =====================================
df_train = pd.read_csv(handout_path+'/train_triplets.txt')
df_test = pd.read_csv(handout_path+'/test_triplets.txt')

# =======================
# construct training data
# =======================
header = []
for i in range(1000):
    header.append('A_feature'+str(i+1))
    header.append('B_feature'+str(i+1))
    header.append('C_feature'+str(i+1))

# comment
df_train_features = pd.DataFrame(columns=header)
df_train_labels = pd.DataFrame(columns=['label'])
i = 1
iterations = len(df_train.values)
for triplet in df_train.values:
    triplet = [int(img) for img in triplet[0].split(' ')]
    # for each triplet, we can construct two possible outputs by switching image B and C
    row1 = np.c_[df.loc[triplet[0]].values,df.loc[triplet[1]].values,df.loc[triplet[2]].values]
    row2 = np.c_[df.loc[triplet[0]].values,df.loc[triplet[2]].values,df.loc[triplet[1]].values]
    row1 = [item for sublist in row1 for item in sublist]
    row2 = [item for sublist in row2 for item in sublist]
    df_train_features.loc[str(i)] = row1
    df_train_labels.loc[str(i)] = [1]
    i += 1
    df_train_features.loc[i] = row2
    df_train_labels.loc[i] = [0]
    i += 1
    print(i/iterations*0.5)

df_train_features.to_csv(handout_path+'train_features.csv')
df_train_labels.to_csv(handout_path+'train_labels.csv')

# ===================
# construct test data
# ===================
df_test_features = pd.DataFrame(columns=header)
i = 1
iterations = len(df_test.values)
for triplet in df_test.values:
    triplet = [int(img) for img in triplet[0].split(' ')]
    row = np.c_[df.loc[triplet[0]].values,df.loc[triplet[1]].values,df.loc[triplet[2]].values]
    row = [item for sublist in row for item in sublist]
    df_test_features.loc[i] = row
    i += 1
    print(i/iterations)

df_test_features.to_csv(handout_path+'test_features.csv')

# ====================
# create NN classifier
# ====================
model = binaryClassification()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.BCELoss()
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
    train_data = trainData(torch.FloatTensor(X_train),
                        torch.FloatTensor(y_train))
    test_data = testData(torch.FloatTensor(X_test))
    train_loader = DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCHSIZE, shuffle=False)
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
        acc = accuracy_score(torch.round(prediction), labels.unsqueeze(1))
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
        y_pred = torch.round(prediction)
        y_pred_list.append(y_pred.cpu().numpy())               

# =======================
# save predictions to csv
# =======================
y_pred_list = [batch.squeeze().tolist() for batch in y_pred_list]   
y_pred_list = np.concatenate(y_pred_list).ravel().tolist()           
np.savetxt("predictions.csv", y_pred_list, fmt="%i")

    