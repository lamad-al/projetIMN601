# -*- coding: utf-8 -*-
"""
Created on Thu Sep 01 17:39:05 2016

@author: Pierre-Marc Jodoin
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier

X = np.r_[2.5*(np.random.randn(50, 2) - [0.2, 1.7]), 2.5*np.random.randn(50, 2) + [1.0, 2.7]]
Y = [0] * 50 + [1] * 50

#########################
###   Perceptron   ######
#########################
classifier = SGDClassifier(loss='perceptron', alpha=0.0, learning_rate='constant', eta0=1, n_iter=100)

# function fit() trains the model
clf = classifier.fit(X, Y)

#
# get the weight of the linear function and
# convert : w0 + w1*x + w2*y=0 into y = mx + b => y = x*w1/w2 - w0/w2
# This is to show the decision boundary
#
w0 = clf.intercept_[0]
w1 = clf.coef_[0][0]
w2 = clf.coef_[0][1]

x = np.linspace(-6, 6, 20)
y = -x*(w1)/(w2)-w0/w2

# plot the decision boundary
plt.figure()
plt.plot(x, y)

# plot the training data
plt.scatter(X[:, 0], X[:, 1], c=Y, s=60, alpha=0.8)

# plot region of class 1 and class 0
xx, yy = np.meshgrid(np.arange(-10, 10, 0.1),np.arange(-10, 10, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)

plt.title("Perceptron ")
plt.show()

predictedY = clf.predict(X)
diff = predictedY - Y
trainingAccuracy = 100*(diff == 0).sum()/np.float(len(Y))
print('Perceptron training accuracy = ',trainingAccuracy, '%')

#################################
###   Logistic regression  ######
#################################

classifier = SGDClassifier(loss='log',alpha=0.001,learning_rate='optimal', eta0=1, n_iter=1000)

# function fit() trains the model
clf = classifier.fit(X, Y)

#
# get the weight of the linear function and
# convert : w0 + w1*x + w2*y=0 into y = mx + b => y = x*w1/w2 - w0/w2
# This is to show the decision boundary
#
w0 = clf.intercept_[0]
w1 = clf.coef_[0][0]
w2 = clf.coef_[0][1]

x=np.linspace(-6,6,20)
y = -x*(w1)/(w2)-w0/w2

# plot the decision boundary
plt.figure();
plt.plot(x,y)

# plot the training data
plt.scatter(X[:, 0], X[:, 1], c=Y, s=60, alpha=0.8)

# plot region of class 1 and class 0
xx, yy = np.meshgrid(np.arange(-10, 10, 0.1),np.arange(-10, 10, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)

plt.title("Sigmoid ")
plt.show()

predictedY = clf.predict(X);
diff = predictedY - Y
trainingAccuracy = 100*(diff == 0).sum()/np.float(len(Y))
print('Logistic regression training accuracy = ',trainingAccuracy, '%')


#################################
###          SVM           ######
#################################
classifier2 = SVC(kernel='linear', C=0.001)

# function fit() trains the model
clf = classifier.fit(X, Y)

#
# get the weight of the linear function and
# convert : w0 + w1*x + w2*y=0 into y = mx + b => y = x*w1/w2 - w0/w2
# This is to show the decision boundary
#
w0 = clf.intercept_[0]
w1 = clf.coef_[0][0]
w2 = clf.coef_[0][1]

x=np.linspace(-6,6,20)
y = -x*(w1)/(w2)-w0/w2

# plot the decision boundary
plt.figure();
plt.plot(x,y)

# plot the training data
plt.scatter(X[:, 0], X[:, 1], c=Y, s=60, alpha=0.8)

# plot region of class 1 and class 0
xx, yy = np.meshgrid(np.arange(-10, 10, 0.1),np.arange(-10, 10, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)

plt.title("SVM ")
plt.show()

predictedY = clf.predict(X);
diff = predictedY - Y
trainingAccuracy = 100*(diff == 0).sum()/np.float(len(Y))
print('SVM training accuracy = ',trainingAccuracy, '%')


#################################
###   3-Class Perceptron   ######
#################################

X = np.r_[(np.random.randn(20, 2) - [1.9, 0.7]), np.random.randn(20, 2) + [1.9, 1.7] , np.random.randn(20, 2) + [1.6, -1.9]]
Y = [0] * 20 + [1] * 20 + [2] * 20


# Perceptron
classifier = SGDClassifier(loss='perceptron',alpha=0.0,learning_rate='constant', eta0=1, n_iter=100)
clf = classifier.fit(X, Y)

xx, yy = np.meshgrid(np.arange(-10, 10, 0.1),np.arange(-10, 10, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure();
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=60, alpha=0.8)
plt.title("3 class Perceptron")
plt.show()

predictedY = clf.predict(X);
diff = predictedY - Y
trainingAccuracy = 100*(diff == 0).sum()/np.float(len(Y))
print('3-Class perceptron training accuracy = ',trainingAccuracy, '%')

