"""
Introduction to Machine Learning
Task 4: Image taste classification
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
May, 2021

TODOs
- assign the images to the triplets
-
"""

import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time
import seaborn as sns
import sys
import random

from tqdm import tqdm

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import ResNet50
from keras import layers
from keras import Model

from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet

# better save than sorry
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# get TensorFlow version
print('TensorFlow version:', tf.__version__)


# Globals
WIDTH = 224
HEIGHT = 224
NUM_CHANNELS = 3
TARGET_SHAPE = (WIDTH, HEIGHT)

IMAGE_PATH = '../handout/food_test/'
TRAIN_TRIPLETS_PATH = '../handout/train_triplets_test.txt'
TEST_TRIPLETS_PATH = '../handout/test_triplets.txt'

SPLIT = 0.7  # split training:validation with 70:30

LEARNING_RATE = 0.002
BATCH_SIZE = 256
EPOCHS = 100


# def load_filenames(path):
#     file_names = sorted(os.listdir(path))
#     return [file for file in file_names if file.endswith('.jpg')]
#
#
# def load_image(filename):
#     image = load_img(filename, color_mode='rgb', target_size=TARGET_SHAPE)
#     image = img_to_array(image, dtype='float32')
#     image = tf.keras.applications.resnet50.preprocess_input(image, data_format='channels_last')
#     return image

def preprocess_image(filename):
    image_string = tf.io.read_file(IMAGE_PATH+filename+'.jpg')
    image = tf.image.decode_jpeg(image_string, channels=NUM_CHANNELS)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, TARGET_SHAPE)
    return image


def preprocess_triplets(anchor, positive, negative):
    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative),
    )


# load triplets
train_triplets = pd.read_csv(TRAIN_TRIPLETS_PATH, names=['A', 'B', 'C'], sep=' ')
test_triplets = pd.read_csv(TEST_TRIPLETS_PATH, names=['A', 'B', 'C'], sep=' ')

# make triplet images a 5 digit number each
for column in ["A", "B", "C"]:
    train_triplets[column] = train_triplets[column].astype(str)
    test_triplets[column] = test_triplets[column].astype(str)
    train_triplets[column] = train_triplets[column].apply(lambda x: x.zfill(5))
    test_triplets[column] = test_triplets[column].apply(lambda x: x.zfill(5))
print(train_triplets)
print(test_triplets)

# assign each image in triplet the image data
img_array = list()
for trip in train_triplets[1:]:
    imgs = preprocess_triplets(trip[0], trip[1], trip[2])
    img_array.append(imgs)








