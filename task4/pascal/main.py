"""
Introduction to Machine Learning
Task 4: Image taste classification
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
May, 2021
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# improve pandas
import numba

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

# better save than sorry
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

#######################
LOAD_PREPROCESSED = True
LOAD_PREPARED_TRAINING_DATA = False
LEARNING_RATE = 0.0002
BATCHSIZE = 64
EPOCHS = 10
#######################

# set random seed
np.random.seed(42)


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


class TrainData(Dataset):
    """ Data loader for training data. """

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TestData(Dataset):
    """ Data loader for test data. """

    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


# Architecture
class BinaryClassification(nn.Module):
    """ Dlass to define NN architecture. """

    def __init__(self):
        super(BinaryClassification, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(len(X_train[0]), 256),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        prediction = self.classifier2(inputs)
        return prediction


def get_image(filename):
    global handout_path
    img = load_img(handout_path + '/food/' + filename, target_size=(224, 224))
    img = img_to_array(img)
    return img


def plot_heatmap(data):
    sns.heatmap(data)
    plt.show()


def transpose_classes(probaA, probaB, probaC):
    return np.concatenate(np.array([probaA, probaB, probaC]).T)


# TODO
def test_construction():
    return 0


# =======================================================================================
# IDEA: PREDICT CLASS LABELS FOR EACH IMAGE INDEPENDENTLY.
# THEN, WE SAVE THE 1000-DIM OUTPUT VECTOR OF EACH IMAGE AS A NEW FEATURE VECTOR.
# WE THEN TRAIN A NEURAL NETWORK WITH 3x1000 INPUTS AND ONE SINGLE OUTPUT
# TO PREDICT A TASTE SIMILARITY BETWEEN THE TWO INPUTS IN RANGE [0,1].
# =======================================================================================

handout_path = "../handout/"

# =============
# PREPROCESSING
# =============
if not LOAD_PREPROCESSED:
    print('Preprocessing images...')

    # setup model
    # we use the pretrained model
    model = tf.keras.Sequential()
    model = ResNet50(include_top=True)
    model.summary()

    # predict image classes (preprocessing)
    start = time.time()
    header = []
    for i in range(1000):
        header.append('class' + str(i + 1))

    df = pd.DataFrame(columns=header)
    imagenames = sorted(os.listdir(handout_path + 'food/'))
    for filename in tqdm(sorted(os.listdir(handout_path + 'food/'))):
        if filename.endswith('.jpg'):
            print('\n' + filename)
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
            print('%s (%.2f%%)' % (label[1], label[2] * 100))
            # append the probabilities to dataframe
            df.loc[filename[0:-4]] = yhat[0][:]

    df.to_csv(handout_path + 'class_probabilities.csv')
    print('elapsed time \t', time.time() - start)

else:
    # load preprocessed images
    df = pd.read_csv(handout_path + 'class_probabilities.csv', index_col=0)

# ===============
# CLASS REDUCTION
# ===============
# print('Reduce classes...')
# start = time.time()
#
# # heat map of class probabilities
# plot_heatmap(df)
#
# # only use classes which are more present a certain threshold in images
# class_mean = df.mean(axis=0)
# threshold = class_mean.mean() + 0.2 * class_mean.std()  # TODO: tune this
# reduced_classes = [idx for idx, c in enumerate(class_mean) if c > threshold]
# print(len(reduced_classes))
#
# # prepare data frame with reduced classes
# df_reduced = df.iloc[:, reduced_classes]
# df_reduced.columns = ['class' + str(i + 1) for i in range(len(reduced_classes))]
# plot_heatmap(df_reduced)
# df_reduced.to_csv(handout_path + 'reduced_class_probabilities.csv')
#
# # clean up
# df = df_reduced
#
# print('elapsed time \t', time.time() - start)

df = pd.read_csv(handout_path + 'reduced_class_probabilities.csv', index_col=0)

# ================
# PREPARE TRIPLETS
# ================
if not LOAD_PREPARED_TRAINING_DATA:
    print('Prepare training data...')
    start = time.time()

    # load the triplets
    df_train = pd.read_csv(handout_path + '/train_triplets_test.txt', header=None)
    df_test = pd.read_csv(handout_path + '/test_triplets_test.txt', header=None)

    # get numpy representation for reduced class probabilities
    classes = df.to_numpy()

    # construct header
    header = list()
    # header.extend(['A_feature'+str(i+1) for i in range(len(df.columns))])
    # header.extend(['B_feature'+str(i+1) for i in range(len(df.columns))])
    # header.extend(['C_feature'+str(i+1) for i in range(len(df.columns))])
    for i in range(len(df.columns)):
        header.append('A_feature'+str(i+1))
        header.append('B_feature'+str(i+1))
        header.append('C_feature'+str(i+1))

    # =======================
    # construct training data
    # =======================
    # assign for each triplet its probability to each of the main classes
    features = np.ndarray(shape=(2 * len(df_train), 3 * len(classes[0])))
    iter = 0
    for triplet in tqdm(df_train.to_numpy()):
        # get image ids per triple
        imgs = [int(img) for img in triplet[0].split(' ')]

        # for each triplet, we can construct two possible outputs by switching image B and C
        # pair of rows denotes that A it closer to B (1) and that A is closer to C (0)

        # # ordering A1 A2 A3 ...
        # features[iter, :] = np.concatenate(classes[[imgs[0], imgs[1], imgs[2]]])
        # features[iter + 1, :] = np.concatenate(classes[[imgs[0], imgs[2], imgs[1]]])

        # ordering A1 B1 C1 ...
        features[iter, :] = transpose_classes(classes[imgs[0]], classes[imgs[1]], classes[imgs[2]])
        features[iter+1, :] = transpose_classes(classes[imgs[0]], classes[imgs[2]], classes[imgs[1]])
        iter += 2

    # transform array to pandas dataframe
    df_train_features = pd.DataFrame(features)
    df_train_features.columns = header
    df_train_features.reset_index(drop=True, inplace=True)
    print(df_train_features.head())

    # label is 1 if A is closer to B and 0 if A is closer to C
    # pair of rows denotes that A is closer to B (1) and that A is closer to C (0)
    df_train_labels = pd.DataFrame(np.where(np.arange(2*len(df_train.values)) % 2, 0, 1), columns=['label'])
    print(df_train_labels.head())

    # write prepared data to csv file
    df_train_features.to_csv(handout_path + 'train_features.csv', index=False)
    df_train_labels.to_csv(handout_path + 'train_labels.csv', index=False)

    # TODO: add test to check whether construction is as intended

    # ===================
    # construct test data
    # ===================
    features = np.ndarray(shape=(len(df_test), 3 * len(classes[0])))
    iter = 0
    for triplet in tqdm(df_test.to_numpy()):
        # get image ids per triple
        imgs = [int(img) for img in triplet[0].split(' ')]

        # construct test data
        # features[iter, :] = np.concatenate(classes[[imgs[0], imgs[1], imgs[2]]])
        features[iter, :] = transpose_classes(classes[imgs[0]], classes[imgs[1]], classes[imgs[2]])
        iter += 1

    # transform test data array to pandas dataframe
    df_test_features = pd.DataFrame(features)
    df_test_features.columns = header
    print(df_test_features.head())

    # write prepared data to csv file
    df_test_features.to_csv(handout_path + 'test_features.csv', index=False)

    # TODO: add test to check whether construction is as intended

    print('elapsed time \t', time.time() - start)

else:
    # load constructed data
    df_train_features = pd.read_csv(handout_path + 'train_features.csv')
    df_train_labels = pd.read_csv(handout_path + 'train_labels.csv')
    df_test_features = pd.read_csv(handout_path + 'test_features.csv')

# ====================
# create NN classifier
# ====================
# model = BinaryClassification()
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
#
# # TODO: maybe need another preprocessing for the data?
# X_train = df_train_features.values
# y_train = df_train_labels.values
# X_test = df_test_features.values

# =================
# run training loop
# =================
# model.train()
# train_results = {}
# test_results = {}
# # iterate over all epochs
# for epoch in tqdm(range(1, EPOCHS + 1)):
#     time_ = AverageMeter()
#     loss_ = AverageMeter()
#     acc_ = AverageMeter()
#     # ===================
#     # Define data loaders
#     # ===================
#     train_data = TrainData(torch.FloatTensor(X_train),
#                            torch.FloatTensor(y_train))
#     test_data = TestData(torch.FloatTensor(X_test))
#     train_loader = DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True)
#     test_loader = DataLoader(dataset=test_data, batch_size=BATCHSIZE, shuffle=False)
#     # iterate over training minibatches
#     for i, data, in enumerate(train_loader, 1):
#         # Accounting
#         end = time.time()
#         features, labels = data
#         features = features.to(device)
#         labels = labels.to(device)
#         bs = features.size(0)
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#         # forward propagation
#         prediction = model(features)
#         # compute the loss
#         loss = criterion(prediction, labels.unsqueeze(1))
#         # Backward propagation
#         loss.backward()
#         # Optimization step.
#         optimizer.step()
#
#         # Accounting
#         acc = accuracy_score(torch.round(prediction), labels.unsqueeze(1))
#         loss_.update(loss.mean().item(), bs)
#         acc_.update(acc.item(), bs)
#         time_.update(time.time() - end)
#
#     print(f'Epoch {epoch}. [Train] \t Time {time_.sum:.2f} Loss \
#             {loss_.avg:.2f} \t Accuracy {acc_.avg:.2f}')
#     train_results[epoch] = (loss_.avg, acc_.avg, time_.avg)
#
# # plot training process
# training = list()
# for key, values in train_results.items():
#     training.append([key, values[0], values[1], values[2]])
# training = list(map(list, zip(*training)))
#
# fig, axs = plt.subplots(2)
# fig.suptitle('Loss and accuracy per epoch')
# axs[0].plot(training[0], training[1], 'b')
# axs[0].set_ylabel('loss')
# axs[0].grid()
# axs[1].plot(training[0], training[2], 'b')
# axs[1].set_ylabel('accuracy')
# axs[1].set_xlabel('epoch')
# axs[1].grid()
# plt.show()

# ===================
# run prediction loop
# ===================
# y_pred_list = []
# model.eval()
# for data in tqdm(test_loader):
#     with torch.no_grad():
#         features = data.to(device)
#         prediction = model(features)
#         y_pred = torch.round(prediction)
#         y_pred_list.append(y_pred.cpu().numpy())

# =======================
# save predictions to csv
# =======================
# y_pred_list = [batch.squeeze().tolist() for batch in y_pred_list]
# y_pred_list = np.concatenate(y_pred_list).ravel().tolist()
# np.savetxt("predictions.csv", y_pred_list, fmt="%i")
