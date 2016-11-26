from keras.datasets import cifar10, mnist
import matplotlib.pyplot as plt

(X_data, X_label), (Y_data, Y_label) = mnist.load_data()
plt.imshow(X_data[1,:,:])

(X_data, X_label), (Y_data, Y_label) = cifar10.load_data()
img = X_data[1,:,:,:]
img = img.transpose(1,2,0)
plt.figure()
plt.imshow(img)

plt.show()