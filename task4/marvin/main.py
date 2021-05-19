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
# example of using a pre-trained model as a classifier
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model



def get_image(filename):
    global handout_path
    img = load_img(handout_path+'/food/'+filename, target_size=(224, 224))
    img = img_to_array(img)
    return img

# IDEA: PREDICT CLASS LABELS FOR EACH IMAGE INDEPENDENTLY.
# THEN, WE SAVE THE 1000-DIM OUTPUT VECTOR OF EACH IMAGE AS A NEW FEATURE VECTOR.
# WE THEN TRAIN A NEURAL NETWORK WITH 2x1000 INPUTS AND ONE SINGLE OUTPUT 
# TO PREDICT A TASTE SIMILARITY BETWEEN THE TWO INPUTS IN RANGE [0,1].


# =========
# load data
# =========
handout_path = "/home/marvin/Downloads/IML_handout_task4/"
# load an image from file
df_train = pd.read_csv(handout_path+'/train_triplets.txt')
df_test = pd.read_csv(handout_path+'/test_triplets.txt')

# ===========
# setup model
# ===========
# we use the pretrained model and change the input and output layers
# the input layer includes 3 stacked images with 3 layers each
# the output is a scalar in range [0,1]
model = tf.keras.Sequential()
model = ResNet50(include_top=True)
model.summary()


# predict image classes
header = []
for i in range(1000):
    header.append(str(i+1))
    
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
        df.loc[filename] = yhat[0][:]

df.to_csv(handout_path+'class_probabilities.csv')