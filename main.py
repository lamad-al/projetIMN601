from keras.datasets import mnist, cifar10
from sklearn.linear_model import SGDClassifier
import numpy as np


########################################################################################################################
#                                                Data
########################################################################################################################
class Images:
    def __init__(self, database=mnist):
        # TODO : Ajouter le slice directement à l'initialisation
        # TODO : Couper un % pour les données de validation plutôt qu'une valeur fixe (afin de s'adapter si on slice)
        """Object used to facilitate the extraction of our data sets and features. Initialize with the database.

        :param database: The database we want to work with (mnist or cifar10)
        """
        # Fetching the data from "keras.datasets"
        assert database in (mnist, cifar10)
        (self.training_and_validation_data, self.training_and_validation_labels),\
        (self.test_data, self.test_labels) = database.load_data()

        # Validation data set - Sub-sample of the training data. We take 5000 samples
        self.validation_data = self.training_and_validation_data[:5000]
        self.validation_labels = self.training_and_validation_data[:5000]

        # Training data set - The remaining samples after taking the validation set
        self.training_data = self.training_and_validation_data[5000:]
        self.training_labels = self.training_and_validation_labels[5000:]

    def get_data_set(self, data_set, feature):
        """Return all the data samples for a given data set with the selected features already extracted.

        :param data_set: The type of data set ('training', 'validation' or 'test')
        :type data_set: str
        :param feature: The type of feature to extract ('gray_scale')
        :type feature: str
        :return: A list with one list of feature for every image in the data set
        :rtype: list
        """
        # Check our input
        assert data_set in ('training', 'validation', 'test')
        assert feature in 'gray_scale'

        # Choose the good data set
        data = {
            'training': self.training_data,
            'validation': self.validation_data,
            'test': self.test_data
        }[data_set]

        # Extract the features
        result = {
            'gray_scale': self.gray_scale(data)
        }[feature]

        return result

    def get_labels(self, data_set):
        """Return all labels (real class of each image) for a given data set.

        :param data_set: The type of data set ('training', 'validation' or 'test')
        :type data_set: str
        :return:
        """
        # Check our input
        assert data_set in ('training', 'test')

        # Choose the good dataset
        labels = {
            'training': self.training_labels,
            'test': self.test_labels
        }[data_set]

        # Transform into a list of integers
        result = []
        for lbl in labels:
            if isinstance(lbl, np.ndarray):
                # Cas de cifar10
                result.append(lbl[0])
            else:
                # Cas de mnist
                result.append(lbl)
        return result

    ####################################################################################################################
    #                                              Features
    ####################################################################################################################
    def gray_scale(self, all_img):
        gs_data = []
        for img in all_img:
            new_img = []
            for line in img:
                if isinstance(line[0], np.ndarray):
                    for RGB in line:
                        gray_val = 0
                        for color_val in RGB:
                            gray_val += color_val
                        new_img.append(gray_val/len(RGB))
                else:
                    for pixel in line:
                        new_img.append(pixel)
            gs_data.append(new_img)
        return gs_data


########################################################################################################################
#                                             Algorithms
########################################################################################################################
def create_confusion_matrix(predicted, real):
    nb_of_class = len(set(real))
    matrix = np.zeros((nb_of_class, nb_of_class), dtype=np.int32)
    for it in range(len(real)):
        matrix[predicted[it]][real[it]] += 1
    return matrix

########################################################################################################################
#                                               Main
########################################################################################################################
if __name__ == '__main__':
    images = Images(database=cifar10)
    X = images.get_data_set(data_set="training", feature="gray_scale")
    Y = images.get_labels(data_set="training")

    ####################################################################################################################
    #                                      Logistic regression
    ####################################################################################################################

    classifier = SGDClassifier(loss='log', alpha=0.001, learning_rate='optimal', eta0=1, n_iter=1000)

    # function fit() trains the model
    clf = classifier.fit(X, Y)

    confusion_matrix = create_confusion_matrix(clf.predict(X), Y)
    print(confusion_matrix)

    diff = clf.predict(X) - Y
    trainingAccuracy = 100 * (diff == 0).sum() / np.float(len(Y))
    print('Logistic regression training accuracy = ', trainingAccuracy, '%')

