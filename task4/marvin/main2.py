
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm


# pytorch
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Lambda, ReLU, Dropout, BatchNormalization
from tensorflow.keras import optimizers, Sequential, Input, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split


####################################
LOAD_PREPROCESSED = True
LOAD_PREPARED_TRAINING_DATA = False
####################################
CHANNELS = 3
WIDTH = 224
HEIGHT = 224
####################################
BATCHSIZE = 64
EPOCHS = 1
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 128
####################################
HANDOUT_PATH = "/home/marvin/Downloads/IML_handout_task4/"  
####################################

# set random seed
np.random.seed(42)

###########################################################################
################################ FUNCTIONS ################################
###########################################################################
#preprocessing and loading the dataset
class SiameseDataset(tf.keras.utils.Sequence):
    def __init__(self,training_csv=None,training_dir=None,width=224,height=224):
        # used to prepare the labels and images path
        self.train_df=pd.read_csv(os.path.join(training_dir, training_csv), sep=' ', dtype=str)
        self.train_df.columns =["A","B","C"]
        self.train_dir = training_dir    
        self.size = (width,height)

    def __getitem__(self,index):
        # getting the image path
        imageA_path=os.path.join(self.train_dir+'food/',str(self.train_df.iat[index,0]))+'.jpg'
        imageB_path=os.path.join(self.train_dir+'food/',str(self.train_df.iat[index,1]))+'.jpg'
        imageC_path=os.path.join(self.train_dir+'food/',str(self.train_df.iat[index,2]))+'.jpg'
        # Loading the image
        imgA = load_img(imageA_path,target_size=self.size)
        imgB = load_img(imageB_path,target_size=self.size)
        imgC = load_img(imageC_path,target_size=self.size)
        # convert to array
        imgA = img_to_array(imgA)
        imgB = img_to_array(imgB)
        imgC = img_to_array(imgC)
        # reshape data for the model
        imgA = imgA.reshape((1, imgA.shape[0], imgA.shape[1], imgA.shape[2]))
        imgB = imgB.reshape((1, imgB.shape[0], imgB.shape[1], imgB.shape[2]))
        imgC = imgC.reshape((1, imgC.shape[0], imgC.shape[1], imgC.shape[2]))
        # prepare the image for the model
        imgA = preprocess_input(imgA)
        imgB = preprocess_input(imgB)
        imgC = preprocess_input(imgC)

        return tf.stack([imgA, imgB, imgC], axis=0), np.array(0)
        
    def __len__(self):
        #return len(self.train_df)
        return len(self.train_df.index.tolist()) // BATCHSIZE


#create a siamese network
class SiameseNetwork():
    '''define a siamese network'''
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Setting up the Sequential of CNN Layers
        basis_inputs = Input(shape=(3, HEIGHT, WIDTH, CHANNELS))
        conv_net = MobileNetV2(include_top=False, input_shape=(HEIGHT, WIDTH, CHANNELS))
        conv_net.trainable=False
        # replace the last fully connected layer
        # the output of the fully connected layer is the embedding of the image (128 components)

        # add new classifier layers
        classifier = Sequential([
            GlobalAveragePooling2D(),
            Dropout(0.5),
            BatchNormalization(),
            Dense(EMBEDDING_DIM),
            Dropout(0.5),
            BatchNormalization(),
            Dense(EMBEDDING_DIM),
            Lambda(lambda x: tf.math.l2_normalize(x, axis=1))])
        
        classifier.trainable = True

        anchor_feature = classifier(conv_net(basis_inputs[:, 0, ...]))
        positive_feature = classifier(conv_net(basis_inputs[:, 1, ...]))
        negative_feature = classifier(conv_net(basis_inputs[:, 2, ...]))
        embeddings = tf.stack([anchor_feature, positive_feature, negative_feature], axis=-1)

        self.model = Model(inputs=basis_inputs, outputs=embeddings)


    def forward_once(self, inputs):
        # Forward pass 
        embedding = self.model(inputs)
        return embedding

    def forward(self, inputs):
        #(input1, input2, input3) = inputs
        # forward pass of input 1
        #output1 = self.forward_once(input1)
        # forward pass of input 2
        #output2 = self.forward_once(input2)
        # forward pass of input 3
        #output3 = self.forward_once(input3)
        # stack the outputs
        #return tf.stack([output1, output2, output3], axis=0), 1
        return self.model(inputs)

        
class TripletLoss():
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=0.1):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs):
        x0 = inputs[..., 0]
        x1 = inputs[..., 1]
        x2 = inputs[..., 2]
        # euclidian distance A-B
        diff1 = x0 - x1
        dist1 = tf.math.sum(tf.math.pow(diff1, 2), 1)
        dist1 = tf.math.sqrt(dist1)

        # euclidian distance A-C
        diff2 = x0 - x2
        dist2 = tf.math.sum(tf.math.pow(diff2, 2), 1)
        dist2 = tf.math.sqrt(dist2)

        # get the triplet loss
        mdist = self.margin + dist1 - dist2
        dist = tf.math.log(tf.math.add(tf.math.exp(mdist),1))
        loss = tf.math.divide(tf.math.sum(dist), BATCHSIZE)
        return loss

def accuracy(_,embeddings):
    dist_positive = tf.reduce_sum(tf.square(embeddings[..., 0] - embeddings[..., 1]), axis=1)
    dist_negative = tf.reduce_sum(tf.square(embeddings[..., 0] - embeddings[..., 2]), axis=1)
    return tf.reduce_mean(tf.cast(tf.greater_equal(dist_negative, dist_positive), tf.float32))

###########################################################################
############################### MAIN SCRIPT ###############################
###########################################################################

# Load the the dataset from raw image folders
train_dataset = SiameseDataset(training_csv='train_triplets.txt',training_dir=HANDOUT_PATH,
                                                                width=WIDTH,height=HEIGHT)
                                                                  

# Declare Siamese Network
SiamNet = SiameseNetwork()

SiamNet.model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss=TripletLoss, metrics=[accuracy])

# ================
# Perform training
# ================
# Train model
SiamNet.model.fit(train_dataset,steps_per_epoch=100, epochs=EPOCHS,validation_data=train_dataset, validation_steps=BATCHSIZE)

# ================
# Predict test set
# ================
test_dataset = SiameseDataset(training_csv='test_triplets.txt',training_dir=HANDOUT_PATH,
                                        width=224,height=224)

