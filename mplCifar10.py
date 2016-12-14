'''Train a simple NN on the CIFAR10 small images dataset.
'''
from __future__ import print_function
from datasets import Cifar10
from images import Images
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import backend as Kback
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
np.random.seed(1337)  # for reproducibility

# Load dataset
images = Images(Cifar10(), slice=1)

batch_size=32
nb_classes=10
nb_epoch=10
nb_filtres=128

# Input image dimensions
img_rows, img_cols = 32, 32

# Get the training, test and validation dataset
# Training set
X_training = images.get_data_set(data_set="training", feature='none')
y_labels = images.get_labels(data_set="training")

# Test set
X_test = images.get_data_set(data_set="test", feature='none')
y_test_labels= images.get_labels(data_set="test")

# Validation set
X_validation = images.get_data_set(data_set='validation', feature='none')
y_validation_labels = images.get_labels(data_set='validation')

# Convert class vectors to binary class matrices
Y_labels = np_utils.to_categorical(y_labels, nb_classes)
Y_test_labels = np_utils.to_categorical(y_test_labels, nb_classes)
Y_validation_labels = np_utils.to_categorical(y_validation_labels, nb_classes)

# Reduce memory usage by limiting the pixel value to 32 bits
X_training = X_training.astype('float32')
X_test = X_test.astype('float32')
X_validation = X_validation.astype('float32')

# Normalization
X_training /= 255
X_test /= 255
X_validation /= 255

# Reshape the data
X_training = X_training.reshape(X_training.shape[0], 3* img_cols * img_rows)
X_test = X_test.reshape(X_test.shape[0], 3* img_cols * img_rows)
X_validation = X_validation.reshape(X_validation.shape[0], 3* img_cols * img_rows)
input_dim = 3* img_rows * img_cols

print('X_training shape:', X_training.shape)
print(X_training.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Build network
model = Sequential()
model.add(Dense(nb_filtres, input_dim=input_dim))

# Hidden layers
for x in range(0, 4):
    model.add(Dense(nb_filtres, activation='relu'))
    model.add(Dropout(0.25))

# Output layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

#set the optimizer
sgd= SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

model.fit(X_training, Y_labels, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=2, validation_data=(X_validation, Y_validation_labels))
score = model.evaluate(X_test, Y_test_labels, verbose=1)
print('Test score:', score[0])
print('Test accuracy():', score[1])