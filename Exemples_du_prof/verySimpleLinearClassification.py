# -*- coding: utf-8 -*-
"""
Created on Thu Sep 01 17:39:05 2016

@author: Pierre-Marc Jodoin
"""

import numpy as np
from sklearn.linear_model import SGDClassifier

X = np.r_[2.5*(np.random.randn(50, 2) - [0.2, 1.7]), 2.5*np.random.randn(50, 2) + [1.0, 2.7]]
Y = [0] * 50 + [1] * 50

#########################
###   Perceptron   ######
#########################
classifier = SGDClassifier(loss='perceptron', alpha=0.0, learning_rate='constant', eta0=1, n_iter=100)

# function fit() trains the model
clf = classifier.fit(X, Y)

# function predict() convert vectors into class labels
predictedY = clf.predict(X)

# lets measure the accuracy of the predictedY 
diff = predictedY - Y
trainingAccuracy = 100*(diff == 0).sum()/np.float(len(Y))
print('Perceptron training accuracy = ',trainingAccuracy, '%')

#
# get the weights of the linear function w0*x0 + w1*x1 + w2=0 
#
w0 = clf.coef_[0][0]
w1 = clf.coef_[0][1]
w2 = clf.intercept_[0]

print('The perceptron just learned the following parameters : ',w0,w1,w2)

