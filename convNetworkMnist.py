'''Train a simple deep CNN on the Mnist small images dataset.
'''
from __future__ import print_function
from images import Images
from datasets import Mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta
from keras import backend as Kback
import numpy as np
np.random.seed(1337)  # for reproducibility

# Choose how much of the dataset you want to use
images = Images(Mnist(), slice=0.1)

#Set parameters
batch_size = 128
nb_classes = 10
nb_epoch = 10

# number of convolutional filters to use
nb_filters = 64

# input image dimensions
img_rows, img_cols = 28, 28

# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# Get the training, test and validation dataset
# Training set
X_training = images.get_data_set(data_set="training", feature='none')
y_labels = images.get_labels(data_set="training")

# Test set
X_test = images.get_data_set(data_set="test", feature='none')
y_test_labels = images.get_labels(data_set="test")

# Validation set
X_validation = images.get_data_set(data_set='validation', feature='none')
y_validation_labels = images.get_labels(data_set='validation')

# Reduce memory usage by limiting the pixel value to 32 bits
X_training = X_training.astype('float32')
X_test = X_test.astype('float32')
X_validation = X_validation.astype('float32')

# Normalization
X_training /= 255
X_test /= 255
X_validation /= 255

# Make sure the image is ordered correctly if the backend is not theano
if Kback.image_dim_ordering() == 'th':
    X_training = X_training.reshape(X_training.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_validation = X_validation.reshape(X_validation.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_training = X_training.reshape(X_training.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_validation = X_validation.reshape(X_validation.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    
print('X_training shape:', X_training.shape)
print(X_training.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
Y_labels = np_utils.to_categorical(y_labels, nb_classes)
Y_test_labels = np_utils.to_categorical(y_test_labels, nb_classes)
Y_validation_labels = np_utils.to_categorical(y_validation_labels, nb_classes)


# Convolution Network2D
model = Sequential()

# Input layer
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))

# Hidden layers
for x in range(0, 3):
    # Layer
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))

    # Pool
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

# Output layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

ada = Adadelta(1.0, rho=0.95, epsilon=1e-08)

model.compile(loss='categorical_crossentropy',
              optimizer=ada,
              metrics=['accuracy'])

model.fit(X_training, Y_labels, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=2, validation_data=(X_validation, Y_validation_labels))
score = model.evaluate(X_test, Y_test_labels, verbose=1)
print('Test score:', score[0])
print('Test accuracy():', score[1])