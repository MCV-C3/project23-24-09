import cv2
import matplotlib.pyplot as plt
import numpy as np


class Detector():
    def __init__(self, detector, standardizer=None):
        self.type = type(detector)
        self.detector = detector
        self.standardizer = standardizer  # StandardScaler(), MinMaxScaler()

    def set_params(self, **parameters):
        """
        Set the parameters of the descriptor
        :param parameters: parameters to set
        :return: self
        """
        # To be overwritten by the child class
        print("Error: set_params not implemented")
        pass

    def get_detector_type(self):
        """
        Get the type of the descriptor
        :return: self.type, type of descriptor
        """
        return self.type

    def get_descriptor(self, filename):
        """
        Compute the descriptor of an image
        :param filename: filename of the image
        :return: keypoints and descriptors of the image
        """
        # To be overwritten by the child class
        print("Error: __get_descriptor not implemented")
        pass

    def compute_descriptors(self, filenames, train=True):
        """
        Compute the descriptors of a list of images
        :param filenames: list of filenames of the images
        :return: list of descriptors
        """
        # print(self.detector.getThreshold())
        if type(filenames) is list or type(filenames) is tuple:
            descriptors_list = []
            keypoints_list = []
            standardized_descriptors = None
            for filename in filenames:
                keypoints, descriptor = self.get_descriptor(filename)
                keypoints_list.append(keypoints)
                descriptors_list.append(descriptor)
            if self.standardizer is not None and train:
                standardized_descriptors = self.standardizer.fit_transform(np.vstack(descriptors_list))
            elif self.standardizer is not None and not train:
                standardized_descriptors = self.standardizer.transform(np.vstack(descriptors_list))
            if standardized_descriptors is not None:
                descriptors_list = np.split(standardized_descriptors, np.cumsum([arr.shape[0] for arr in descriptors_list])[:-1])
            return keypoints_list, descriptors_list
        elif type(filenames) is str:
            keypoints, descriptor = self.get_descriptor(filenames)
            return keypoints, descriptor
        else:
            print("Wrong type of filenames, must be list or string")
            return None

    def draw_keypoints(self, img, keypoints):
        """
        Draw keypoints on an image
        :param img: image
        :param keypoints: keypoints
        :return: image with keypoints drawn
        """
        image = cv2.drawKeypoints(img, keypoints, 0, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.figure(figsize=(20,10))
        plt.imshow(image)
        plt.show()
        return image


class KazeDetector(Detector):
    def __init__(self, threshold=0.0001, standardizer=None):
        """
        Constructor of the KazeDetector class
        :param threshold: Detector threshold
        """
        super().__init__(cv2.KAZE_create(threshold=threshold), standardizer)
        self.threshold = threshold

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['detector']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.detector = cv2.KAZE_create(threshold=state['threshold'])

    def set_params(self, **parameters):
        """
        Set the parameters of the descriptor
        :param parameters: parameters to set
        """
        if 'threshold' in parameters:
            self.detector.setThreshold(parameters['threshold'])
            print("Changing threshold to", self.detector.getThreshold())
            self.threshold = parameters['threshold']
        if 'stand' in parameters:
            self.standardizer = parameters['stand']

    def get_descriptor(self, filename):
        """
        Compute the descriptor of an image
        :param filename: filename of the image
        :return: keypoints and descriptors of the image
        """
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        kpt, des = self.detector.detectAndCompute(gray, None)
        return kpt, des


class DenseSiftDetector(Detector):
    def __init__(self, step=10, scale=10, standardizer=None):
        """
        Constructor of the DenseSiftDetector class
        :param step: step between two keypoints
        :param scale: scale of the keypoints
        """
        super().__init__(cv2.xfeatures2d.SIFT_create(), standardizer)
        self.step = step
        self.scale = scale

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['detector']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.detector = cv2.xfeatures2d.SIFT_create()

    def set_params(self, **parameters):
        """
        Set the parameters of the descriptor
        :param parameters: parameters to set
        """
        if 'step' in parameters:
            self.step = parameters['step']
        if 'scale' in parameters:
            self.scale = parameters['scale']
        if 'contrast_thresh' in parameters:
            self.detector.setContrastThreshold(parameters['contrast_thresh'])
        if 'edge_thresh' in parameters:
            self.detector.setEdgeThreshold(parameters['edge_thresh'])
        if 'stand' in parameters:
            self.standardizer = parameters['stand']

    def get_descriptor(self, filename):
        """
        Compute the descriptor of an image
        :param filename: filename of the image
        :return: keypoints and descriptors of the image
        """
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)
        keypoints = [cv2.KeyPoint(x, y, self.scale) for y in range(0, gray.shape[0], self.step) for x in range(0, gray.shape[1], self.step)]
        kpt, des = self.detector.compute(gray, keypoints)
        return kpt, des