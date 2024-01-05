import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import pairwise_kernels
from sklearn import preprocessing

def histogram_intersection_kernel(X, Y):
    """
    Histogram intersection kernel to be used in SVM
    :param X: first matrix
    :param Y: second matrix
    :return: histogram intersection kernel
    """
    min_values = np.minimum(X[:, np.newaxis, :], Y)
    intersection_array = np.sum(min_values, axis=2)
    return intersection_array


class Classifier():
    def __init__(self, classifier_type, classifier):
        self.type = classifier_type
        self.classifier = classifier

    def train(self, descriptors, labels):
        """
        Train the classifier
        :param descriptors: descriptors of the images
        :param labels: labels of the images
        """
        # To be overwritten by the child class
        pass

    def set_params(self, **params):
        """
        Set parameters
        :param params: parameters
        """
        self.classifier.set_params(**params)

    def predict(self, descriptor):
        """
        Predict the label of an image
        :param descriptor: descriptor of the image
        :return: predicted label
        """
        # To be overwritten by the child class
        pass

    def score(self, descriptors, labels):
        """
        Score of the classifier
        :param descriptors: descriptors of the images
        :param labels: labels of the images
        :return: score of the classifier
        """
        # To be overwritten by the child class
        pass


class KNNClassifier(Classifier):
    def __init__(self, k, distance_metric='euclidean'):
        """
        Constructor of the KNeighborsClassifier class
        :param k: number of neighbors
        """
        classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, metric=distance_metric, probability=True)
        super().__init__("KNN", classifier=classifier)

    def set_params(self, **params):
        """
        Set parameters
        :param params: parameters
        """
        self.classifier.set_params(**params)
        if 'metric' in params:
            print("Metric changed to " + params['metric'])
        if 'n_neighbors' in params:
            print("Number of neighbors changed to " + str(params['n_neighbors']))

    def train(self, descriptors, labels):
        """
        Train the classifier
        :param descriptors: descriptors of the images
        :param labels: labels of the images
        """
        print("Training KNN classifier...")
        self.classifier.fit(descriptors, labels)

    def predict(self, descriptor):
        """
        Predict the label of an image
        :param descriptor: descriptor of the image
        :return: predicted label
        """
        return self.classifier.predict(descriptor)
    
    def predict_proba(self, descriptor):
        """
        Predict the label of an image
        :param descriptor: descriptor of the image
        :return: predicted label
        """
        return self.classifier.predict_proba(descriptor)

    def score(self, descriptors, labels):
        """
        Score of the classifier
        :param descriptors: descriptors of the images
        :param labels: labels of the images
        :return: score of the classifier
        """
        return self.classifier.score(descriptors, labels)


class SVMClassifier(Classifier):
    def __init__(self, kernel='linear'):
        """
        Constructor of the SVMClassifier class
        :param kernel: kernel of the SVM
        """
        if kernel == 'histogram_intersection':
            classifier = SVC(kernel=histogram_intersection_kernel, probability=True)
        else:
            classifier = SVC(kernel=kernel, probability=True)
        super().__init__("SVM", classifier=classifier)

    def set_params(self, **params):
        """
        Set parameters
        :param params: parameters
        """
        if 'kernel' in params:
            print("Kernel changed to " + params['kernel'])
            if params['kernel'] == 'histogram_intersection':
                self.classifier = SVC(kernel=histogram_intersection_kernel)
            else:
                self.classifier = SVC(kernel=params['kernel'])

    def train(self, descriptors, labels):
        """
        Train the classifier
        :param descriptors: descriptors of the images
        :param labels: labels of the images
        """
        print("Training SVM classifier...")
        self.classifier.fit(descriptors, labels)

    def predict(self, descriptor):
        """
        Predict the label of an image
        :param descriptor: descriptor of the image
        :return: predicted label
        """
        return self.classifier.predict(descriptor)
    
    def predict_proba(self, descriptor):
        """
        Predict the label of an image
        :param descriptor: descriptor of the image
        :return: predicted label
        """
        return self.classifier.predict_proba(descriptor)

    def score(self, descriptors, labels):
        """
        Score of the classifier
        :param descriptors: descriptors of the images
        :param labels: labels of the images
        :return: score of the classifier
        """
        return self.classifier.score(descriptors, labels)