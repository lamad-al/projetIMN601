from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
import abc


class Classifiers:
    """Abstract class. Represents a certain classifier."""
    __metaclass__ = abc.ABCMeta

    def __init__(self, dataset):
        self.dataset = dataset

    @abc.abstractmethod
    def get_classifier(self):
        """Return the classifier itself (sklearn object) with all the parameters."""
        return

    @abc.abstractmethod
    def get_name(self):
        """Return the name (string) of the classifier. Mostly used for printing or writing in a log."""
        return


class Sigmoid(Classifiers):
    def get_name(self):
        return "Logistic Sigmoid"

    def get_classifier(self):
        return LogisticRegression(n_jobs=-1, solver='sag', multi_class='multinomial')


class Adaboost(Classifiers):
    def get_name(self):
        return "Adaboost"

    def get_classifier(self):
        weak_classifier = SGDClassifier(loss='log', alpha=0.0001, learning_rate='invscaling', eta0=1, n_iter=5,
                                        power_t=0.5, n_jobs=-1)
        if self.dataset.get_name is "mnist":
            return AdaBoostClassifier(weak_classifier, algorithm="SAMME", n_estimators=75, learning_rate=0.000001)
        else:
            return AdaBoostClassifier(weak_classifier, algorithm="SAMME", n_estimators=500, learning_rate=0.5)


########################################################################################################################
#                                            Grid Search
########################################################################################################################
def check_SGDClassifier(X, Y, V, W):
    """Execute a grid search for a Logistic Sigmoid classifier.

    Check for the constant, optimal and inverse scaling learning rate.
    The results are printed such as we can copy them into a spreadsheet.
    constant : alpha ; eta0 ; n_iter ; accuracy (%)
    optimal : alpha ; n_iter ; accuracy (%)
    invscaling : alpha ; eta0 ; n_iter ; power_t ; accuracy (%)

    :param X: A training data set with the features already extracted.
    :param Y: The labels (real classes) of the training data set.
    :param V: A validation data set with the features already extracted.
    :param W: The labels (real classes) of the validation data set.
    :return: Nothing. Print the information on the output.
    """
    best_accuracy = 0
    print("Grid Search for SGDClassifier")
    # Constant
    print("Learning Rate --- Constant")
    for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
        for eta0 in [0.1, 0.5, 1, 5, 10, 100]:
            for n_iter in [1, 5, 10, 50, 100, 500, 1000]:
                clf = SGDClassifier(loss='log', alpha=alpha, learning_rate='constant', eta0=eta0, n_iter=n_iter,
                                    n_jobs=-1)
                clf.fit(X, Y)
                accuracy = clf.score(V, W) * 100
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                print("{};{};{};{}"
                      .format(alpha, eta0, n_iter, accuracy))
    # Optimal
    print("Learning Rate --- Optimal")
    for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
        for n_iter in [1, 5, 10, 50, 100, 500, 1000]:
            clf = SGDClassifier(loss='log', alpha=alpha, learning_rate='optimal', n_iter=n_iter,
                                n_jobs=-1)
            clf.fit(X, Y)
            accuracy = clf.score(V, W) * 100
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            print("{};{};{}"
                  .format(alpha, n_iter, accuracy))
    # InvScaling
    print("Learning Rate --- InvScaling")
    for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
        for eta0 in [0.1, 0.5, 1, 5, 10, 100]:
            for n_iter in [1, 5, 10, 50, 100, 500, 1000]:
                for power_t in [0.1, 0.3, 0.5, 0.9]:
                    clf = SGDClassifier(loss='log', alpha=alpha, learning_rate='invscaling', eta0=eta0, n_iter=n_iter,
                                        power_t=power_t, n_jobs=-1)
                    clf.fit(X, Y)
                    accuracy = clf.score(V, W) * 100
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                    print("{};{};{};{};{}"
                          .format(alpha, eta0, n_iter, power_t, accuracy))
    print("Best accuracy was : ", best_accuracy, "%")


def check_Adaboost(X, Y, V, W):
    """Execute a grid search for Adaboost with a Logistic Sigmoid classifier.

    The results are printed such as we can copy them into a spreadsheet.
    n_estimators ; learning_rate ; accuracy (%)

    :param X: A training data set with the features already extracted.
    :param Y: The labels (real classes) of the training data set.
    :param V: A validation data set with the features already extracted.
    :param W: The labels (real classes) of the validation data set.
    :return: Nothing. Print the information on the output.
    """
    print("Grid Search for Adaboost")
    for n_estimators in [1, 5, 10, 25, 50, 75, 100, 500, 1000]:
        for learning_rate in [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1]:
            weak_classifier = SGDClassifier(loss='log', alpha=0.0001, learning_rate='invscaling', eta0=1, n_iter=5,
                                            power_t=0.5, n_jobs=-1)
            clf = AdaBoostClassifier(base_estimator=weak_classifier, algorithm='SAMME',
                                     n_estimators=n_estimators, learning_rate=learning_rate)
            clf.fit(X, Y)
            print("{};{};{}".format(n_estimators, learning_rate, clf.score(V, W) * 100))

