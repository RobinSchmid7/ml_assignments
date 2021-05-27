"""
Introduction to Machine Learning
Task 4: Image taste classification
Team Naiveoutliers
Robin Schmid, Pascal Mueller, Marvin Harms
May, 2021
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Lambda
from tensorflow.keras import optimizers, Sequential, Input
from tensorflow.keras.applications import Xception
from sklearn.model_selection import train_test_split

WIDTH = 96
HEIGHT = 96
CHANNELS = 3

VALIDATION_SIZE = 0.15
NUM_EMBEDDINGS = 300
LEARNING_RATE = 0.001
EPOCHS = 2

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
BUFFER_SIZE = 1024

HANDOUT_PATH = '../handout/'

# Preprocessing
def create_split_data():
    with open(HANDOUT_PATH + 'train_triplets.txt', 'r') as file:
        triplets = [l for l in file.readlines()]

    train_data, val_data = train_test_split(triplets, test_size=VALIDATION_SIZE, random_state=42)

    with open(HANDOUT_PATH + 'train_data.txt', 'w') as file:
        for e in train_data:
            file.write(e)
    with open(HANDOUT_PATH + 'val_data.txt', 'w') as file:
        for e in val_data:
            file.write(e)

def load_dataset(filename):
    data = tf.data.TextLineDataset(filename)
    data = data.map(lambda triplet: load_triplets(triplet),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Returns a tf tensor containing all triplets of the file, each element is a image tensor
    return data

def load_triplets(triplet):
    idx = tf.strings.split(triplet)
    anchor = process_image(HANDOUT_PATH + '/food/' + idx[0] + '.jpg')
    positive = process_image(HANDOUT_PATH + '/food/' + idx[1] + '.jpg')
    negative = process_image(HANDOUT_PATH + '/food/' + idx[2] + '.jpg')
    return tf.stack([anchor, positive, negative], axis=0)

def process_image(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (HEIGHT, WIDTH))
    return img

def create_model(num_embeddings):
    # Pretrained model
    basis_inputs = Input(shape=(3, HEIGHT, WIDTH, CHANNELS))
    basis_model = Xception(include_top=False, input_shape=(HEIGHT, WIDTH, CHANNELS))
    # Do not train pretrained model
    basis_model.trainable = False

    # Additional layers for prediction
    extended_model = Sequential([
        GlobalAveragePooling2D(),
        Dense(num_embeddings),
        Lambda(lambda x: tf.math.l2_normalize(x, axis=1))])

    # Create embeddings
    anchor_embedding = extended_model(basis_model(basis_inputs[:, 0, ...]))
    positive_embedding = extended_model(basis_model(basis_inputs[:, 1, ...]))
    negative_embedding = extended_model(basis_model(basis_inputs[:, 2, ...]))
    embeddings = tf.stack([anchor_embedding, positive_embedding, negative_embedding], axis=-1)

    # Siamese model
    siamese_model = tf.keras.Model(inputs=basis_inputs, outputs=embeddings)
    # siamese_model.summary()

    return siamese_model

def create_prediction_model(model):
    dist_positive = tf.reduce_sum(tf.square(model.output[..., 0] - model.output[..., 1]), axis=1)
    dist_negative = tf.reduce_sum(tf.square(model.output[..., 0] - model.output[..., 2]), axis=1)
    predictions = tf.cast(tf.greater_equal(dist_negative, dist_positive), tf.int8)
    return tf.keras.Model(inputs=model.inputs, outputs=predictions)

def triplet_loss(_,embeddings):
    dist_positive = tf.reduce_sum(tf.square(embeddings[..., 0] - embeddings[..., 1]), axis=1)
    dist_negative = tf.reduce_sum(tf.square(embeddings[..., 0] - embeddings[..., 2]), axis=1)
    return tf.reduce_mean(tf.math.softplus(dist_positive - dist_negative))

def accuracy(_,embeddings):
    dist_positive = tf.reduce_sum(tf.square(embeddings[..., 0] - embeddings[..., 1]), axis=1)
    dist_negative = tf.reduce_sum(tf.square(embeddings[..., 0] - embeddings[..., 2]), axis=1)
    return tf.reduce_mean(tf.cast(tf.greater_equal(dist_negative, dist_positive), tf.float32))


if __name__ == '__main__':
    # Load data
    create_split_data()
    train_data = load_dataset(HANDOUT_PATH + 'train_data.txt')
    val_data = load_dataset(HANDOUT_PATH + 'val_data.txt')

    # Number of training and test data
    with open(HANDOUT_PATH + 'train_data.txt', 'r') as file:
        num_train_data = sum(1 for line in file)
    with open(HANDOUT_PATH + 'test_triplets.txt', 'r') as file:
        num_test_data = sum(1 for line in file)

    # Create model
    model = create_model(NUM_EMBEDDINGS)
    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), loss=triplet_loss, metrics=[accuracy])
    # model.summary()

    # Load batches
    train_data = train_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True).repeat().batch(TRAIN_BATCH_SIZE)
    val_data = val_data.batch(TRAIN_BATCH_SIZE)

    # Train model
    model.fit(train_data, steps_per_epoch=int(np.ceil(num_train_data / TRAIN_BATCH_SIZE)),
              epochs=EPOCHS,validation_data=val_data,validation_steps=10)

    test_data = load_dataset(HANDOUT_PATH + 'test_triplets.txt')
    test_data = test_data.batch(TEST_BATCH_SIZE).prefetch(2)
    prediction_model = create_prediction_model(model)

    predictions = prediction_model.predict(test_data,steps=int(np.ceil(num_test_data / TEST_BATCH_SIZE)),verbose=1)
    np.savetxt('predictions3.txt', predictions, fmt='%i')

    print("Finished!")