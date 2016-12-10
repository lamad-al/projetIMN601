from images import Images
from datasets import Mnist, Cifar10
from classifiers import Sigmoid, Adaboost, check_SGDClassifier, check_lbp
from sklearn.metrics import confusion_matrix


def img_classification():
    """Main algorithm"""
    # Choose which databases to use
    datasets = (Mnist(), Cifar10(),)[0:2]

    # Choose which classifiers to use
    classifiers = (Sigmoid(), Adaboost(),)[0:1]

    # Choose which features to use
    features = ["raw_pixels", "lbp", "gray_scale"][1:2]

    for dataset in datasets:
        for feature in features:
            for classifier in classifiers:
                images = Images(dataset, slice=0.1)
                # Get the training data set with its labels
                X = images.get_data_set(data_set="training", feature=feature)
                Y = images.get_labels(data_set="training")

                # Train the classifier
                # TODO : Ajout d'un autre feature
                clf = classifier.get_classifier().fit(X, Y)

                # Get the accuracy for the training data set
                # TODO : Le message est horrible... Le rendre beau svp
                print(classifier.get_name(), '\'s training accuracy for', dataset.get_name(),
                      'with', feature, 'for feature = ',
                      clf.score(X, Y) * 100, '%')

                # Get the test data set with its labels
                X = images.get_data_set(data_set="test", feature=feature)
                Y = images.get_labels(data_set="test")

                # Get the accuracy for the test data set
                print(classifier.get_name(), '\'s test accuracy for', dataset.get_name(),
                      'with', feature, 'for feature = ',
                      clf.score(X, Y) * 100, '%')

                # Print a confusion matrix
                cm = confusion_matrix(Y, clf.predict(X))
                print(cm)


def execute_grid_search():
    """Execute a grid search. Is separated from the main algorithm because it was just too long to execute."""
    # Choose which databases to use
    datasets = (Mnist(), Cifar10(),)[0:1]

    # Choose which features to use
    features = ["raw_pixels", "lbp"][1:2]

    for dataset in datasets:
        for feature in features:
            images = Images(dataset, slice=0.1)
            # Get the training data set with its labels
            X = images.get_data_set(data_set="training", feature=feature)
            Y = images.get_labels(data_set="training")

            # Get the validation data set with its labels
            V = images.get_data_set(data_set="validation", feature=feature)
            W = images.get_labels(data_set="validation")

            # Choose which grid search to execute
            check_SGDClassifier(X, Y, V, W)
            #check_lbp(X, Y, V, W, p, r)

########################################################################################################################
#                                               Main
########################################################################################################################
if __name__ == '__main__':
    # Choose if we want to execute a grid search or the main algorithm
    grid_search = False

    if grid_search:
        execute_grid_search()
    else:
        img_classification()






