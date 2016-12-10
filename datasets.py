from keras.datasets import mnist, cifar10
import abc


class Dataset:
    """Abstract class. Represents a certain data set."""
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_dataset(self):
        """Return the keras.datasets data set."""
        return

    @abc.abstractmethod
    def get_name(self):
        """Return the name (string) of the data set. Mostly for printing and writing in a log."""
        return

    @abc.abstractmethod
    def get_dimension(self):
        """Return the dimension (1D for mnist, 3D for cifar10). Mostly for the feature extraction."""
        return


class Cifar10(Dataset):
    def get_dimension(self):
        return 3

    def get_dataset(self):
        return cifar10

    def get_name(self):
        return "cifar10"


class Mnist(Dataset):
    def get_dimension(self):
        return 1

    def get_dataset(self):
        return mnist

    def get_name(self):
        return "mnist"
