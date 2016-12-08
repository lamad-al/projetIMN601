from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
import abc


class Classifiers:
    # TODO : Adding parameters with Grid Search
    """Abstract class. Represents a certain classifier."""
    __metaclass__ = abc.ABCMeta

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
        return SGDClassifier(loss='log', alpha=0.001, learning_rate='optimal', eta0=1, n_iter=1000)
    