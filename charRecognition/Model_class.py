import numpy as np
import pandas as pd
import cv2
import os
import glob
from preprocess import *

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import RMSprop

IMAGE_SIZE = (12, 28)
BATCH_SIZE = 128
EPOCHS = 50
MAX_HEIGHT = 28
MAX_WIDTH = 12

ALPHA_DICT = {'0' : 0, '1' : 1, '2' : 2, '3' : 3, '4' : 4, '5' : 5, '6' : 6, '7' : 7, '8' : 8, '9' : 9, 'A' : 10, 'B' : 11, 'C' : 12, 'D' : 13,
              'E': 14, 'F': 15, 'G': 16, 'H': 17, 'K': 18, 'L': 19, 'M': 20, 'N': 21, 'P': 22, 'R': 23, 'S' : 24, 'T' : 25, 'U' : 26,
              'V' : 27, 'X' : 28, 'Y' : 29, 'Z' : 30}

def load_images_from_mul_folder(folder):
    images = []
    rls = []
    for s_folder in os.listdir(folder):
        for filename in os.listdir(folder + "/" + s_folder):
            img = cv2.imread(os.path.join(folder + "/" + s_folder, filename))
            rls.append(ALPHA_DICT[s_folder])
            if img is not None:
                i = ProcessImage(img)
                i = cv2.bitwise_not(i)
                i = np.expand_dims(i, axis=-1)
                images.append(i)
    return images, rls

train, y_train = load_images_from_mul_folder('./charRecognition/charTrainset/')
train = np.array(train, dtype = 'float32')

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(train)
class CNN_Model(object):
    def __init__(self, trainable=True):
        self.batch_size = BATCH_SIZE
        self.trainable = trainable
        self.num_epochs = EPOCHS
        # Building model
        self._build_model()

        # Input data
        if trainable:
            self.model.summary()
            self.data = train
            self.y = y_train

        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['acc'])

    def _build_model(self):
        # CNN model
        self.model = Sequential()
        self.model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(28, 12, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(31, activation='softmax'))

    def train(self):
        learning_rate_reduction = ReduceLROnPlateau(monitor='acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

        print("Training......")
        trainX = self.data
        trainY = self.y
        trainX = np.array(trainX)
        trainY = np.array(trainY)

        self.model.fit(datagen.flow(trainX, trainY, batch_size=self.batch_size), callbacks=[learning_rate_reduction], verbose=1,
                       epochs=self.num_epochs, shuffle=True)
    def save(self, filename):
        self.model.save(filename)
