from math import ceil
from mahotas.colors import rgb2grey
from mahotas.features.lbp import lbp, lbp_transform
from skimage.feature import local_binary_pattern


class Images:
    def __init__(self, Dataset, slice=1):
        """Object used to facilitate the extraction of our data sets and features. Initialize with the database.

        :param database: The database we want to work with (mnist or cifar10)
        :param slice: Number between ]0,1] which represents the % of data that we want for training. Default: 100%
        :type slice: float
        """
        assert 0 < slice <= 1

        self.dataset = Dataset

        # Fetching the data from "keras.datasets"
        (self.training_and_validation_data, self.training_and_validation_labels),\
        (self.test_data, self.test_labels) = Dataset.get_dataset().load_data()

        # Evaluate how many training samples we must keep and slice accordingly
        nb_samples = ceil(len(self.training_and_validation_data) * slice)
        nb_validation_samples = ceil(nb_samples * 0.1)
        self.training_and_validation_data = self.training_and_validation_data[:nb_samples]
        self.training_and_validation_labels = self.training_and_validation_labels[:nb_samples]

        # Validation data set - Sub-sample of the training data. We take 10% of the training data set
        self.validation_data = self.training_and_validation_data[:nb_validation_samples]
        self.validation_labels = self.training_and_validation_labels[:nb_validation_samples]

        # Training data set - The remaining samples after taking the validation set
        self.training_data = self.training_and_validation_data[nb_validation_samples:]
        self.training_labels = self.training_and_validation_labels[nb_validation_samples:]

        # Print a cute little message
        # TODO : Faire en sorte de pouvoir afficher plusieurs noms si on ajoute des db
        print("Loaded {} training data, {} validation data and {} test data for {}."
              .format(len(self.training_data),
                      len(self.validation_data),
                      len(self.test_data),
                      Dataset.get_name()))

    def get_data_set(self, data_set, feature):
        """Return all the data samples for a given data set with the selected features already extracted.

        :param data_set: The type of data set ('training', 'validation' or 'test')
        :type data_set: str
        :param feature: The type of feature to extract ('gray_scale', 'raw_pixels', 'rgb')
        :type feature: str
        :return: A list with one list of feature for every image in the data set
        :rtype: list
        """
        # Check our input
        assert data_set in ('training', 'validation', 'test')
        assert feature in ('gray_scale', 'raw_pixels', 'lbp', 'none')

        # Choose the good data set
        data = {
            'training': self.training_data,
            'validation': self.validation_data,
            'test': self.test_data
        }[data_set]

        if feature is None:
            return data

        # Extract the features
        function = {
            'gray_scale': self.gray_scale,
            'raw_pixels': self.raw_pixels,
            'lbp': self.local_binary_patterns
        }[feature]
        result = function(data)

        return result

    def get_labels(self, data_set):
        """Return all labels (real class of each image) for a given data set.

        :param data_set: The type of data set ('training', 'validation' or 'test')
        :type data_set: str
        :return: A list of all labels (real class of each image) for a given data set.
        :rtype: list
        """
        # Check our input
        assert data_set in ('training', 'validation', 'test')

        # Choose the good dataset
        labels = {
            'training': self.training_labels,
            'validation': self.validation_labels,
            'test': self.test_labels
        }[data_set]

        # Transform into a list of integers
        result = []
        for lbl in labels:
            if self.dataset.get_dimension() != 1:
                result.append(lbl[0])
            else:
                result.append(lbl)
        return result

    ####################################################################################################################
    #                                              Features
    ####################################################################################################################
    def gray_scale(self, all_img):
        """Transform a list of images into a list of gray scale feature.

        :param all_img: A data set returned by "keras.datasets"
        :return: A list with one list of feature (gray scale) for every image.
        """
        gs_data = []
        if self.dataset.get_dimension() == 1:
            for img in all_img:
                new_img = []
                for line in img:
                    for pixel in line:
                        new_img.append(pixel)
                gs_data.append(new_img)
        else:
            for img in all_img:
                new_img = []
                conversion = rgb2grey(img)
                for line in conversion:
                    for pixel in line:
                        new_img.append(pixel)
                gs_data.append(new_img)
        return gs_data

    def raw_pixels(self, all_img):
        """Every image becomes a list of raw pixels (no other transformation).

        For RGB images, for a pixel p, the representation is: p[1].red, p[1].green, p[1].blue, p[2].red, ...

        :param all_img: A data set returned by "keras.datasets"
        :return: A list with one list of feature (one pixel after the other) for every image.
        """
        gs_data = []
        for img in all_img:
            new_img = []
            for line in img:
                if self.dataset.get_dimension() > 1:
                    for RGB in line:
                        for color_val in RGB:
                            new_img.append(color_val)
                else:
                    for pixel in line:
                        new_img.append(pixel)
            gs_data.append(new_img)
        return gs_data

    def rgb_splitter(self, img):
        new_img = []
        for line in img:
            new_line = []
            for RGB in line:
                for pixel in RGB:
                    new_line.append(pixel)
            new_img.append(new_line)
        return new_img

    def local_binary_patterns(self, all_img):
        lbp_data = []
        for img in all_img:
            if self.dataset.get_dimension() > 1:
                img = self.rgb_splitter(img)
            new_img = []
            for line in local_binary_pattern(img, P=8, R=10, method='uniform'):
                for pixel in line:
                    new_img.append(pixel)
            lbp_data.append(new_img)
        return lbp_data
