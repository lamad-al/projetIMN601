from images import Images
from datasets import Mnist, Cifar10
from classifiers import Sigmoid
from sklearn.metrics import confusion_matrix


########################################################################################################################
#                                               Main
########################################################################################################################
if __name__ == '__main__':
    # TODO : Automatiser les features
    # Choose which databases to use
    datasets = (Mnist(), Cifar10(),)[0:1]

    # Choose which classifiers to use
    classifiers = (Sigmoid(),)[0:1]

    # Choose which features to use
    features = ["raw_pixels", "gray_scale"][0:1]

    for dataset in datasets:
        images = Images(dataset, slice=0.1)

        for feature in features:
            # Get the training data set with its labels
            X = images.get_data_set(data_set="training", feature=feature)
            Y = images.get_labels(data_set="training")

            # Get the validation data set with its labels
            V = images.get_data_set(data_set="validation", feature=feature)
            W = images.get_labels(data_set="validation")

            for classifier in classifiers:
                # Get the parameters with Grid Search
                # TODO : Faire une liste des paramètres à trouver et setter
                # TODO : Implémenter un Grid Search (regarder si Keras en a pas un, ça serait trop nice...)

                # Train the classifier
                # TODO : Ajout d'un autre algorithme et d'un autre feature
                # TODO : Utiliser nos paramètres trouvés par Grid Search
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




