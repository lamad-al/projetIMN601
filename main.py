from images import Images
from datasets import Mnist, Cifar10
from classifiers import Sigmoid, Adaboost, check_SGDClassifier, check_Adaboost
from sklearn.metrics import confusion_matrix


def img_classification():
    """Main algorithm"""
    # Choose which databases to use
    datasets = (Mnist(), Cifar10(),)[0:1]

    # Choose which classifiers to use
    classifiers = (Sigmoid, Adaboost,)[0:2]

    # Choose which features to use
    features = ["raw_pixels", "hog", "lbp", "gray_scale"][0:2]

    for dataset in datasets:
        for feature in features:
            for classifier in classifiers:
                classifier = classifier(dataset)
                images = Images(dataset, slice=1)
                # Get the training data set with its labels
                X = images.get_data_set(data_set="training", feature=feature)
                Y = images.get_labels(data_set="training")

                # Train the classifier
                print("Training {} --- {} --- {}".format(dataset.get_name(), classifier.get_name(), feature))
                clf = classifier.get_classifier().fit(X, Y)

                # Get the accuracy for the training data set
                print("Training Accuracy: {}%".format(clf.score(X, Y) * 100))

                # Get the test data set with its labels
                X = images.get_data_set(data_set="test", feature=feature)
                Y = images.get_labels(data_set="test")

                # Get the accuracy for the test data set
                print("Test Accuracy: {}%".format(clf.score(X, Y) * 100))

                # Print a confusion matrix
                cm = confusion_matrix(Y, clf.predict(X))
                print(cm)


def execute_grid_search():
    """Execute a grid search. Is separated from the main algorithm because it was just too long to execute."""
    # Choose which databases to use
    datasets = (Mnist(), Cifar10(),)[0:2]

    # Choose which features to use
    features = ["raw_pixels", "hog"][0:2]

    for dataset in datasets:
        print("For data set: {}".format(dataset.get_name()))
        for feature in features:
            print("For feature: {}".format(feature))
            images = Images(dataset, slice=0.1)
            # Get the training data set with its labels
            X = images.get_data_set(data_set="training", feature=feature)
            Y = images.get_labels(data_set="training")

            # Get the validation data set with its labels
            V = images.get_data_set(data_set="validation", feature=feature)
            W = images.get_labels(data_set="validation")

            # Choose which grid search to execute
            #check_SGDClassifier(X, Y, V, W)
            check_Adaboost(X, Y, V, W)

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






