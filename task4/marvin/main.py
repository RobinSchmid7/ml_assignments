"""
Introduction to Machine Learning
Task 4: Image taste classification
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
May, 2021

Version 1.2

Usage: python main.py --eps 100 --bsz 64 --lr 0.002

# TODO: adding more classses only using resnet does hardly improve it, use multiple pretrained networks and only use the most common classes (related to food)...
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
import argparse

# example of using a pre-trained model as a classifier
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

####################################
LOAD_PREPROCESSED = True
LOAD_PREPARED_TRAINING_DATA = True
N_FEATURES = 134
####################################
LEARNING_RATE = 0.001
BATCHSIZE = 64
EPOCHS = 30
####################################
HANDOUT_PATH = "/home/marvin/Downloads/IML_handout_task4/"  
####################################

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


class BinaryClassification(nn.Module):
    """ Class to define NN architecture. """

    def __init__(self):
        super(BinaryClassification, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(len(X_train[0]), 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.classifier(inputs)


def load_images(model, df):
    imagenames = sorted(os.listdir(HANDOUT_PATH + 'food/'))
    for filename in tqdm(sorted(os.listdir(HANDOUT_PATH + 'food/'))):
        if filename.endswith('.jpg'):
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
            # append the probabilities to dataframe
            df.loc[filename[0:-4]] = yhat[0][:]
    return df


def get_image(filename):
    global HANDOUT_PATH
    img = load_img(HANDOUT_PATH + '/food/' + filename, target_size=(224, 224))
    img = img_to_array(img)
    return img


def plot_heatmap(data):
    sns.heatmap(data)
    plt.show()


def prepare_training_features(class_proba, df_features):
    """ Modifies the train triplets such that image is assigned a
    probability to belong to a class.

    The training triplets are arrange such that image A is always
    closer to image B than to C. To increase the number of features
    the output of each triplet is additionally modified, i.e. by switching
    image B and C. This results in two feature vectors for each triplet.

    Returns a data frame of size (2 x triplets, 3x classes). The ordering for
    each triplet is:
        triplet 0: A1 B1 C1 ...
        triplet 0: A1 C1 B1 ...
        triplet 1: A1 B1 C1 ...
        triplet 1: A1 C1 B1 ...
    """
    def transpose_classes(probaA, probaB, probaC):
        return np.concatenate(np.array([probaA, probaB, probaC]).T)

    # assign for each triplet its probability to each of the main classes
    features = np.ndarray(shape=(2 * len(df_features), 3 * len(class_proba[0])))
    for i, triplet in enumerate(tqdm(df_features.to_numpy())):
        # get image ids per triple
        imgs = [int(img) for img in triplet[0].split(' ')]

        # for each triplet, we can construct two possible outputs by switching image B and C
        # pair of rows denotes that A it closer to B (1) and that A is closer to C (0)
        # ordering A1 B1 C1 ...
        features[2*i, :] = transpose_classes(class_proba[imgs[0]], class_proba[imgs[1]], class_proba[imgs[2]])
        features[2*i+1, :] = transpose_classes(class_proba[imgs[0]], class_proba[imgs[2]], class_proba[imgs[1]])

    return pd.DataFrame(features)


def prepare_test_features(class_proba, df_features):
    """ Modifies the test triplets such that each image is assigned a
    probability to belong to a class.

    Returns a data frame of size (triplets, 3 x classes). The ordering for each
    triplet is:
        triplet 0: A1 B1 C1 ...
        triplet 1: A1 B1 C1 ...
    """
    def transpose_classes(probaA, probaB, probaC):
        return np.concatenate(np.array([probaA, probaB, probaC]).T)

    features = np.ndarray(shape=(len(df_features), 3 * len(class_proba[0])))
    for i, triplet in enumerate(tqdm(df_features.to_numpy())):
        # get image ids per triple
        imgs = [int(img) for img in triplet[0].split(' ')]
        # construct test data
        features[i, :] = transpose_classes(class_proba[imgs[0]], class_proba[imgs[1]], class_proba[imgs[2]])

    return pd.DataFrame(features)


############## MAIN ###############
if __name__ == '__main__':
    # =================
    # preprocess images
    # =================
    if not LOAD_PREPROCESSED:
        # we use the pretrained model
        model = tf.keras.Sequential()
        model = ResNet50(include_top=True)
        model.summary()

        header = []
        for i in range(1000):
            header.append('class' + str(i + 1))
        
        df_images = pd.DataFrame(columns=header)
        df_images = load_images(model, df_images)
        df_images.to_csv(HANDOUT_PATH + 'class_probabilities.csv')
    
    else:
        df_images = pd.read_csv(HANDOUT_PATH + 'class_probabilities.csv', index_col=0)

    # =========================
    # Construct feature vectors
    # =========================
    if not LOAD_PREPARED_TRAINING_DATA:
        # reduce number of classes
        X = df_images.values

        #standardize data
        scaler = StandardScaler()
        X_standardized  = scaler.fit_transform(X)

        # dimensionality reduction
        pca = PCA(n_components=N_FEATURES)
        X_reduced = pca.fit_transform(X=X_standardized)

        # construct new dataframe
        header = ['feature_'+str(i+1) for i in range(N_FEATURES)]
        df_reduced = pd.DataFrame(index=df_images.index,columns=header,data=X_reduced)
        plot_heatmap(df_images)
        df_reduced.to_csv(HANDOUT_PATH + 'reduced_class_probabilities.csv')

        # load the triplets
        df_train = pd.read_csv(HANDOUT_PATH + '/train_triplets.txt', header=None)
        df_test = pd.read_csv(HANDOUT_PATH + '/test_triplets.txt', header=None)

        # construct header
        header = list()
        for i in range(len(df_reduced.columns)):
            header.append('A_class'+str(i+1))
            header.append('B_class'+str(i+1))
            header.append('C_class'+str(i+1))
        
        classes = df_reduced.values

        # construct training features
        df_train_features = prepare_training_features(classes, df_train)
        df_train_features.columns = header
        df_train_labels = pd.DataFrame(np.where(np.arange(2*len(df_train.values)) % 2, 0, 1), columns=['label'])
        print(df_train_features.head())

        # construct test features
        df_test_features = prepare_test_features(classes, df_test)
        df_test_features.columns = header
        print(df_test_features.head())

        # write prepared data to csv file
        df_train_features.to_csv(HANDOUT_PATH + 'train_features.csv', index=False)
        df_train_labels.to_csv(HANDOUT_PATH + 'train_labels.csv', index=False)
        df_test_features.to_csv(HANDOUT_PATH + 'test_features.csv', index=False)

    
    else:
        df_train_features = pd.read_csv(HANDOUT_PATH + 'train_features.csv')
        df_train_labels = pd.read_csv(HANDOUT_PATH + 'train_labels.csv')
        df_test_features = pd.read_csv(HANDOUT_PATH + 'test_features.csv')

    # ==========
    # split data
    # ==========
    y_train = df_train_labels.values
    X_train = df_train_features.values
    X_test = df_test_features.values

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1, shuffle=True, random_state=42)

    # define model, loss and optimizer
    model = BinaryClassification()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # =================
    # run training loop
    # =================
    model.train()
    train_results = {}
    test_results = {}

    # iterate over all epochs
    for epoch in tqdm(range(1, EPOCHS + 1)):
        time_ = AverageMeter()
        loss_ = AverageMeter()
        acc_ = AverageMeter()

        # define data loader
        train_data = TrainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=False)
        validation_data = TrainData(torch.FloatTensor(X_validation), torch.FloatTensor(y_validation))
        validation_loader = DataLoader(dataset=validation_data, batch_size=100, shuffle=True)

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
            loss = criterion(prediction, labels)

            # backward propagation
            loss.backward()

            # optimization step
            optimizer.step()

            # validate accuracy
            acc = []
            model.eval()
            with torch.no_grad():
                for j, data_validation, in enumerate(validation_loader, 1):
                    if j>3:
                        break
                    features_validation, labels_validation = data_validation
                    features_validation = features_validation.to(device)
                    labels_validation = labels_validation.to(device)
                    prediction_validation = model(features_validation)
                    acc.append(accuracy_score(labels_validation,torch.round(prediction_validation).detach().numpy()))

            model.train()


            # accounting
            acc = np.mean(acc)
            loss_.update(loss.mean().item(), bs)
            acc_.update(acc.item(), bs)
            time_.update(time.time() - end)

        print(f'\n Epoch {epoch}. [Train] \t Time {time_.sum:.2f} \t Loss {loss_.avg:.2f} \t Accuracy {acc_.avg:.2f}')
        train_results[epoch] = (loss_.avg, acc_.avg, time_.avg)

        # nice print to console
        time.sleep(0.1)
    
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
    print('Classify test images...')
    test_data = TestData(torch.FloatTensor(X_test))
    test_loader = DataLoader(dataset=test_data, batch_size=BATCHSIZE, shuffle=False)
    y_pred_list = []
    model.eval()
    for data in tqdm(test_loader):
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
    print('=== Finished prediction ===')
    print('Epochs: {}, Batch size: {}, Learning rate {}'.format(EPOCHS, BATCHSIZE, LEARNING_RATE))

