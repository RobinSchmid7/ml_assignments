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
# example of using a pre-trained model as a classifier
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model



def get_image(number):
    global handout_path
    img = load_img(handout_path+'/food/'+str(number)+'.jpg', target_size=(224, 224))
    img = img_to_array(img)
    return img

def stack_images(img1,img2,img3):
    pass


# =========
# load data
# =========
handout_path = "/home/marvin/Downloads/IML_handout_task4"
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
ResNet = ResNet50(include_top=False, input_shape=(224,224,3))
# mark loaded layers as not trainable
for layer in ResNet.layers:
	layer.trainable = False

# add new classifier layers
model.add(ResNet)
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# define new model
#model = Model(inputs=model.inputs, outputs=output)
# summarize
model.summary()



image = get_image('00001')
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# load the model
# predict the probability across all output classes
yhat = model.predict(image)
# retrieve the most likely result, e.g. highest probability
label = yhat[0][0]
# print the classification
print(yhat)