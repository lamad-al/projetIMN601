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
images = Images(Mnist(), slice=1)

#Set parameters
batch_size = 128
nb_classes = 10
nb_epoch = 10
nb_filters = 512

# input image dimensions
img_rows, img_cols = 28, 28

# Get the training, test and validation dataset
# Training set
X_training = images.get_data_set(data_set="training", feature='none')
y_labels = images.get_labels(data_set="training")

# Test set
X_test = images.get_data_set(data_set="test", feature='none')
y_labels_test = images.get_labels(data_set="test")

# Validation set
X_validation = images.get_data_set(data_set='validation', feature='none')
y_validation_label = images.get_labels(data_set='validation')

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
    X_training = X_training.reshape(X_training.shape[0], img_cols * img_rows)
    X_test = X_test.reshape(X_test.shape[0], img_cols * img_rows)
    X_validation = X_validation.reshape(X_validation.shape[0], img_cols * img_rows)
    input_shape = (1, img_rows, img_cols)
else:
    X_training = X_training.reshape(X_training.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    X_validation = X_validation.reshape(X_validation.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Convert class vectors to binary class matrices
Y_labels = np_utils.to_categorical(y_labels, nb_classes)
Y_labels_test = np_utils.to_categorical(y_labels_test, nb_classes)
Y_validation_label = np_utils.to_categorical(y_validation_label, nb_classes)


# Convolution Network2D
model = Sequential()
model.add(Dense(nb_filters, input_dim=img_cols * img_rows))

# Hidden layers
for x in range(0, 4):
    # Layer
    model.add(Dense(nb_filters, activation='relu'))
    model.add(Dropout(0.25))

# Output layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Classifier
sgd= SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

# Training
model.fit(X_training, Y_labels, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=2, validation_data=(X_validation, Y_validation_label))
score = model.evaluate(X_test, Y_labels_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy():', score[1])
