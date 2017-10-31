from __future__ import print_function

import sys
sys.setrecursionlimit(10000)

import os
import time
import pickle

import densenet
import numpy as np
import sklearn.metrics as metrics

from keras.datasets import cifar100
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K

batch_size = 64
nb_classes = 100
nb_epoch = 15

img_rows, img_cols = 32, 32
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = 12
bottleneck = False
reduction = 0.0
dropout_rate = 0.0 # 0.0 for data augmentation


model = densenet.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                          growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate,
                          bottleneck=bottleneck, reduction=reduction, weights=None)
print("Model created")

model.summary()
optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
print("Finished compiling")
print("Building model...")

# Load model
model_name = "CIFAR100_DenseNet-40-12_1000_epochs.h5"
model.load_weights("weights/" + model_name)
print("Model loaded.")











# (trainX, trainY), (testX, testY) = cifar100.load_data(label_mode='fine')
# trainX = trainX.astype('float32')
# testX = testX.astype('float32')

# trainX /= 255.
# testX /= 255.

# Y_train = np_utils.to_categorical(trainY, nb_classes)
# Y_test = np_utils.to_categorical(testY, nb_classes)






(trainX, trainY), (testX, testY) = cifar100.load_data(label_mode='fine')

with open('./test_data', 'rb') as f:
    test_data = pickle.load(f)

x_test = test_data.reshape(10000,3,32,32).transpose(0,2,3,1)
# x_test = x_test[:1000,:,:,:]
# x_test = x_test[:25,:,:,:]
# testY = testY[:25,:]

x_test = x_test.astype('float32')
x_test /= 255.

t = time.time()
one_hot_test_predictions = model.predict(x_test)
elapsed = time.time() - t
elapsed

not_hot_test_predictions = np.round(np.argmax(one_hot_test_predictions, axis = 1))

predictions_dir = os.path.join(os.getcwd(), 'saved_predictions')
predictions_name = model_name[:-3]
predictions_path = os.path.join(predictions_dir, predictions_name)

if not os.path.isdir(predictions_dir):
    os.makedirs(predictions_dir)

# np.savetxt(predictions_path, not_hot_test_predictions, delimiter=",", format='%d')
not_hot_test_predictions.tofile(predictions_path, sep='\n', format='%d')

# (trainX, trainY), (testX, testY) = cifar100.load_data(label_mode='fine')
# # overwrite from kaggle file
# testX = x_test

# (trainX, trainY), (testX, testY) = cifar100.load_data(label_mode='fine')

# trainX = trainX.astype('float32')
# testX = testX.astype('float32')

# trainX /= 255.
# testX /= 255.

# testX = testX[:25,:,:,:]
# testY = testY[:25]


# yPreds = model.predict(testX)
# yPred = np.argmax(yPreds, axis=1)

yPred = not_hot_test_predictions
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)

