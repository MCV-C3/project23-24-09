import numpy as np
from sklearn.cluster import MiniBatchKMeans

class Codebook():
    def __init__(self, codebook_type, codebook):
        self.type = codebook_type
        self.codebook = codebook
        self.k = codebook.n_clusters

    def set_params(self, **params):
        """
        Set the parameters of the codebook
        :param params: parameters
        """
        self.codebook.set_params(**params)

    def codebook_fit(self, descriptors):
        """
        Compute the codebook
        :param descriptors: descriptors of the images
        :return: codebook
        """
        # To be overwritten by the child class
        pass

    def get_visual_words(self, descriptor):
        """
        Visual words
        :param descriptor: descriptor
        :param k: number of clusters
        :return: visual words
        """
        # To be overwritten by the child class
        pass

    def compute_visual_words(self, descriptors):
        """
        Compute the visual words of a list of descriptors
        :param descriptors: descriptors of the images
        :return: visual words
        """
        if type(descriptors) is list:
            visual_words=np.zeros((len(descriptors),self.k),dtype=np.float32)
            for i in range(len(descriptors)):
                visual_words[i,:] = self.get_visual_words(descriptors[i])
            return visual_words
        elif type(descriptors) is np.ndarray and descriptors.ndim == 2 and descriptors.shape[0] == 1:
            return self.get_visual_words(descriptors)
        else:
            print("Wrong type of descriptors, must be list or numpy array")
            return None


class KmeansCodebook(Codebook):
    def __init__(self, k):
        """
        Constructor of the KmeansCodebook class
        :param k: number of clusters
        """
        super().__init__("KMEANS", MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20,compute_labels=False,reassignment_ratio=10**-4,random_state=42))

    def set_params(self, **params):
        """
        Set the parameters of the codebook
        :param params: parameters
        """
        self.codebook.set_params(**params)
        if "n_clusters" in params:
            print("Changing Kmeans cluster number to " + str(params["n_clusters"]))
            self.k = self.codebook.n_clusters

    def codebook_fit(self, descriptors):
        """
        Compute the codebook with kmeans
        :param descriptors: descriptors of the images
        :return: codebook
        """
        print("Fitting Kmeans codebook...")
        self.codebook.fit(descriptors)

    def get_visual_words(self, descriptor):
        """
        Visual words
        :param descriptor: descriptor
        :param k: number of clusters
        :return: visual words
        """
        if descriptor is not None and descriptor.shape[0] > 0:
            words = self.codebook.predict(descriptor)
            return np.bincount(words, minlength=self.k)
        return np.zeros(self.k)
