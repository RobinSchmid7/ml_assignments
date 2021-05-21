"""
Introduction to Machine Learning
Task 4: Image taste classification
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
May, 2021

Version 1.2

Usage: python main.py --eps 100 --bsz 64 --lr 0.002
"""

import numpy as np
import pandas as pd
# import tensorflow as tf
# import keras
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import argparse

# example of using a pre-trained model as a classifier
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.resnet50 import decode_predictions
# from keras.applications.resnet50 import ResNet50

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PowerTransformer

# better save than sorry
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# DATA LOADING
LOAD_PREPROCESSED = True
LOAD_PREPARED_TRAINING_DATA = True
THRESHOLD_STD = 0.02

# HYPERPARAMETERS
LEARNING_RATE = 0.002
BATCHSIZE = 64
EPOCHS = 100

# PATHS
HANDOUT_PATH = "../handout/"

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
        self.classifier1 = nn.Sequential(
            nn.Linear(len(X_train[0]), 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.classifier2 = nn.Sequential(
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
        return self.classifier1(inputs)


# def load_images(model, df):
#     imagenames = sorted(os.listdir(HANDOUT_PATH + 'food/'))
#     for filename in tqdm(sorted(os.listdir(HANDOUT_PATH + 'food/'))):
#         if filename.endswith('.jpg'):
#             print('\n' + filename)
#             image = get_image(filename)
#             # reshape data for the model
#             image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#             # prepare the image for the model
#             image = preprocess_input(image)
#             # predict the probability across all output classes
#             yhat = model.predict(image)
#             # retrieve the most likely result, e.g. highest probability
#             label = decode_predictions(yhat)
#             # retrieve the most likely result, e.g. highest probability
#             label = label[0][0]
#             print('%s (%.2f%%)' % (label[1], label[2] * 100))
#             # append the probabilities to dataframe
#             df.loc[filename[0:-4]] = yhat[0][:]
#     return df


# def get_image(filename):
#     global HANDOUT_PATH
#     img = load_img(HANDOUT_PATH + '/food/' + filename, target_size=(224, 224))
#     img = img_to_array(img)
#     return img


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
    # assign for each triplet its probability to each of the main classes
    features = np.ndarray(shape=(2 * len(df_features), 3 * len(class_proba[0])))
    for i, triplet in enumerate(tqdm(df_train.to_numpy())):
        # get image ids per triple
        imgs = [int(img) for img in triplet[0].split(' ')]

        # for each triplet, we can construct two possible outputs by switching image B and C
        # pair of rows denotes that A it closer to B (1) and that A is closer to C (0)

        # # ordering A1 A2 A3 ...
        # features[iter, :] = np.concatenate(classes[[imgs[0], imgs[1], imgs[2]]])
        # features[iter + 1, :] = np.concatenate(classes[[imgs[0], imgs[2], imgs[1]]])

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
    features = np.ndarray(shape=(len(df_features), 3 * len(class_proba[0])))
    for i, triplet in enumerate(tqdm(df_features.to_numpy())):
        # get image ids per triple
        imgs = [int(img) for img in triplet[0].split(' ')]

        # construct test data
        # features[iter, :] = np.concatenate(classes[[imgs[0], imgs[1], imgs[2]]])
        features[i, :] = transpose_classes(class_proba[imgs[0]], class_proba[imgs[1]], class_proba[imgs[2]])

    return pd.DataFrame(features)


def transpose_classes(probaA, probaB, probaC):
    """ Transposes an array of ordering
        A1 A2 ... B1 B2 ... C1 C2 ...
    to an array of ordering
        A1 B1 C1 A2 B2 C2 ...
    """
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

if __name__ == '__main__':
    # ===============
    # PARSE ARGUMENTS
    # ===============
    parser = argparse.ArgumentParser(description='Hyperparameters for NN')
    parser.add_argument('--eps', type=int, default=EPOCHS, help='epochs')
    parser.add_argument('--bsz', type=int, default=BATCHSIZE, help='batchsize')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='learning rate')
    args = parser.parse_args()

    # =============
    # PREPROCESSING
    # =============
    if not LOAD_PREPROCESSED:
        print('Preprocessing images...')
        #
        # # setup model
        # # we use the pretrained model
        # model = tf.keras.Sequential()
        # model = ResNet50(include_top=True)
        # model.summary()
        #
        # # predict image classes (preprocessing)
        # start = time.time()
        # header = []
        # for i in range(1000):
        #     header.append('class' + str(i + 1))
        #
        # df_images = pd.DataFrame(columns=header)
        # df_images = load_images(model, df_images)
        #
        # print('...writing class probabilities')
        # df_images.to_csv(HANDOUT_PATH + 'class_probabilities.csv')
        #
        # print('elapsed time \t', time.time() - start)

    else:
        print('Loading class probabilities...')
        # load preprocessed images
        df_images = pd.read_csv(HANDOUT_PATH + 'class_probabilities.csv', index_col=0)

    # ================
    # PREPARE TRIPLETS
    # ================
    if not LOAD_PREPARED_TRAINING_DATA:
        # ===============
        # CLASS REDUCTION
        # ===============
        print('Reduce classes...')

        # heat map of class probabilities
        plot_heatmap(df_images)

        # only use classes which are more present than a certain threshold in images
        class_mean = df_images.mean(axis=0)
        threshold = class_mean.mean() + THRESHOLD_STD * class_mean.std()  # TODO: tune this
        reduced_classes = [idx for idx, c in enumerate(class_mean) if c > threshold]
        print(len(reduced_classes))

        # prepare data frame with reduced classes
        df_reduced = df_images.iloc[:, reduced_classes]
        df_reduced.columns = ['class' + str(i + 1) for i in range(len(reduced_classes))]
        plot_heatmap(df_reduced)
        df_reduced.to_csv(HANDOUT_PATH + 'reduced_class_probabilities.csv')

        df_reduced = pd.read_csv(HANDOUT_PATH + 'reduced_class_probabilities.csv', index_col=0)

        print('Prepare training data...')
        start = time.time()

        # load the triplets
        df_train = pd.read_csv(HANDOUT_PATH + '/train_triplets.txt', header=None)
        df_test = pd.read_csv(HANDOUT_PATH + '/test_triplets.txt', header=None)

        # get numpy representation for reduced class probabilities
        classes = df_reduced.to_numpy()

        # construct header
        header = list()
        # header.extend(['A_class'+str(i+1) for i in range(len(df.columns))])
        # header.extend(['B_class'+str(i+1) for i in range(len(df.columns))])
        # header.extend(['C_class'+str(i+1) for i in range(len(df.columns))])
        for i in range(len(df_reduced.columns)):
            header.append('A_class'+str(i+1))
            header.append('B_class'+str(i+1))
            header.append('C_class'+str(i+1))

        # construct training features
        df_train_features = prepare_training_features(classes, df_train)
        df_train_features.columns = header
        df_train_features.reset_index(drop=True, inplace=True)
        print(df_train_features.head())

        # construct training labels
        # pair of rows denotes that A is closer to B (1) and that A is closer to C (0)
        df_train_labels = pd.DataFrame(np.where(np.arange(2*len(df_train.values)) % 2, 0, 1), columns=['label'])
        print(df_train_labels.head())

        # construct test features
        df_test_features = prepare_test_features(classes, df_test)
        df_test_features.columns = header
        print(df_test_features.head())

        # write prepared data to csv file
        df_train_features.to_csv(HANDOUT_PATH + 'train_features.csv', index=False)
        df_train_labels.to_csv(HANDOUT_PATH + 'train_labels.csv', index=False)
        df_test_features.to_csv(HANDOUT_PATH + 'test_features.csv', index=False)

        print('elapsed time \t', time.time() - start)

        # TODO: add test to check whether construction is as intended

    else:
        print('Loading training and test set...')
        # load constructed data
        df_train_features = pd.read_csv(HANDOUT_PATH + 'train_features.csv')
        df_train_labels = pd.read_csv(HANDOUT_PATH + 'train_labels.csv')
        df_test_features = pd.read_csv(HANDOUT_PATH + 'test_features.csv')

    # ====================
    # CREATE NN CLASSIFIER
    # ====================
    # extract training and test data
    y_train = df_train_labels.values
    X_test = df_test_features.values
    X_train = df_train_features.values

    # TODO: maybe need another preprocessing for the features, e.g PowerTransformer?
    scaler = PowerTransformer()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # define model, loss and optimizer
    model = BinaryClassification()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # =================
    # run training loop
    # =================
    model.train()
    train_results = {}
    test_results = {}
    print('Running training loop...')

    # iterate over all epochs
    for epoch in tqdm(range(1, args.eps + 1)):
        time_ = AverageMeter()
        loss_ = AverageMeter()
        acc_ = AverageMeter()

        # define data loader
        train_data = TrainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(dataset=train_data, batch_size=args.bsz, shuffle=True)

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

            # accounting
            acc = accuracy_score(torch.round(prediction).detach().numpy(), labels)
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
    test_loader = DataLoader(dataset=test_data, batch_size=args.bsz, shuffle=False)
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
    print('Epochs: {}, Batch size: {}, Learning rate {}'.format(args.eps, args.bsz, args.lr))