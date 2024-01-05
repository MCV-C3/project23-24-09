import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score, precision_score, f1_score


class CustomModel(BaseEstimator, ClassifierMixin):
    """
    Custom classifier model that includes a local feature detector, a codebook and a classifier
    """
    def __init__(self, detector, codebook, classifier, dim_reduction=None, spatial_pyramid=None):
        """
        Constructor of the CustomModel class
        :param detector: detector
        :param codebook: codebook
        :param classifier: classifier
        """
        self.detector = detector
        self.codebook = codebook
        self.classifier = classifier
        self.dim_reduction = dim_reduction
        self.spatial_pyramid = None
        if spatial_pyramid is not None:
            if type(spatial_pyramid) is not tuple or len(spatial_pyramid) != 2:
                print("Wrong type of spatial pyramid. Must be a tuple with 2 elements")
                exit()
            else:
                self.spatial_pyramid = spatial_pyramid
        self.classes_ = np.array(['Opencountry', 'coast', 'forest', 'highway', 'inside_city', 'mountain', 'street', 'tallbuilding'])

    def dim_reduction_fit(self, visual_words, y):
        """
        Fit the dimensionality reduction
        :param visual_words: visual words
        :param y: labels
        """
        if type(self.dim_reduction) is PCA:
            return self.dim_reduction.fit_transform(visual_words)
        elif type(self.dim_reduction) is LinearDiscriminantAnalysis:
            return self.dim_reduction.fit_transform(visual_words, y)
        else:
            print("Wrong type of dimensionality reduction. Must be PCA or LDA")
            return None

    def spatial_pyramid_visual_words(self, keypoints_list, descriptors_list):
        """
        Compute the visual words of a list of descriptors using spatial pyramid
        :param keypoints: keypoints of the images (used to know their coordinates in the image)
        :param descriptors: descriptors of the keypoints
        :return: visual words
        """
        descriptors_divided_by_level = {}
        for row in range(self.spatial_pyramid[0]):
            for col in range(self.spatial_pyramid[1]):
                # Size of images is always 256x256, so we calculate the coordinates for each block depending on
                # the level of the pyramid
                min_coord_row = int((256*row)/self.spatial_pyramid[0])
                if row == self.spatial_pyramid[0] - 1:
                    max_coord_row = 256
                else:
                    max_coord_row = int((256*(row + 1))/self.spatial_pyramid[0])
                min_coord_col = int((256*col)/self.spatial_pyramid[1])
                if col == self.spatial_pyramid[1] - 1:
                    max_coord_col = 256
                else:
                    max_coord_col = int((256*(col + 1))/self.spatial_pyramid[1])

                block_descriptors_list = []
                for image_keypoints, image_descriptors in zip(keypoints_list, descriptors_list):
                    if len(image_keypoints) == 0:
                        block_descriptors_list.append(None)
                        continue
                    image_block_descriptors = np.zeros((0, len(image_descriptors[0])), dtype=type(image_descriptors[0][0]))
                    for keypoint, descriptor in zip(image_keypoints, image_descriptors):
                        # Check if the keypoint is inside the block
                        if min_coord_row <= keypoint.pt[1] < max_coord_row and min_coord_col <= keypoint.pt[0] < max_coord_col:
                            image_block_descriptors = np.vstack((image_block_descriptors, descriptor), dtype=type(descriptor[0]))
                    block_descriptors_list.append(image_block_descriptors)
                descriptors_divided_by_level[(row, col)] = block_descriptors_list

        # Compute visual words for each block
        dict_visual_words = {}
        for row in range(self.spatial_pyramid[0]):
            for col in range(self.spatial_pyramid[1]):
                # Compute visual words for each block
                visual_words = self.codebook.compute_visual_words(descriptors_divided_by_level[(row, col)])
                # Add visual words to the dictionary
                dict_visual_words[(row, col)] = visual_words

        # Concatenate visual_words for each image
        visual_words = np.zeros((len(descriptors_list), 0))
        for row in range(self.spatial_pyramid[0]):
            for col in range(self.spatial_pyramid[1]):
                visual_words = np.hstack((visual_words, dict_visual_words[(row, col)]))

        return visual_words

    def set_params(self, **params):
        """
        Set parameters
        :param params: parameters
        """
        params_dict = {'detector': {}, 'codebook': {}, 'classifier': {}, 'dim_reduction': {}, 'spatial_pyramid': {}}
        for param, value in params.items():
            if 'detector' in param:
                params_dict['detector'][param.split('__')[1]] = value
            elif 'codebook' in param:
                params_dict['codebook'][param.split('__')[1]] = value
            elif 'classifier' in param:
                params_dict['classifier'][param.split('__')[1]] = value
            elif 'dim_reduction' in param:
                params_dict['dim_reduction'][param.split('__')[1]] = value
            elif 'spatial_pyramid' in param:
                params_dict['spatial_pyramid'][param.split('__')[1]] = value

        self.detector.set_params(**params_dict['detector'])
        self.codebook.set_params(**params_dict['codebook'])
        self.classifier.set_params(**params_dict['classifier'])
        if self.dim_reduction is not None:
            self.dim_reduction.set_params(**params_dict['dim_reduction'])
        if self.spatial_pyramid is not None:
            if 'levels' in params_dict['spatial_pyramid']:
                if type(params_dict['spatial_pyramid']['levels']) is not tuple or len(params_dict['spatial_pyramid']['levels']) != 2:
                    print("Wrong type of spatial pyramid. Must be a tuple with 2 elements")
                    exit()
                else:
                    self.spatial_pyramid = params_dict['spatial_pyramid']['levels']
        return self

    def calculate_visual_words(self, kpts_list, descriptors_list):
        """
        Compute the visual words of a list of descriptors using spatial pyramid (if any)
        :param kpts_list: list of keypoints of the images
        :param descriptors_list: list of descriptors of the images
        :return: visual words
        """
        if self.spatial_pyramid is None:
            visual_words = self.codebook.compute_visual_words(descriptors_list)
        else:
            visual_words = self.spatial_pyramid_visual_words(kpts_list, descriptors_list)
        return visual_words

    def fit(self, X, y):
        """
        Fit the model
        :param X: descriptors of the images
        :param y: labels of the images
        """
        # Get train descriptors
        train_keypoints, train_descriptors = self.detector.compute_descriptors(X, train=True)
        D = np.vstack(train_descriptors)

        # Get and fit KMeans codebook
        self.codebook.codebook_fit(D)

        # Get visual words
        visual_words = self.calculate_visual_words(train_keypoints, train_descriptors)

        # Get and fit dimensionality reduction (if any)
        if self.dim_reduction is not None:
            visual_words = self.dim_reduction_fit(visual_words, y)

        # Train classifier
        self.classifier.train(visual_words, y)

    def predict(self, X):
        """
        Predict the labels of a list of images
        :param X: Image filenames
        :return: predicted labels
        """
        # Get  descriptors
        kpts, descriptors = self.detector.compute_descriptors(X, train=False)

        # Get visual words
        visual_words = self.calculate_visual_words(kpts, descriptors)

        # Get dimensionality reduction (if any)
        if self.dim_reduction is not None:
            visual_words = self.dim_reduction.transform(visual_words)

        # Predict labels
        return self.classifier.predict(visual_words)
    
    def predict_proba(self, X):
        """
        Predict the labels of a list of images
        :param X: Image filenames
        :return: predicted labels
        """
        # Get  descriptors
        kpts, descriptors = self.detector.compute_descriptors(X, train=False)

        # Get visual words
        visual_words = self.calculate_visual_words(kpts, descriptors)

        # Get dimensionality reduction (if any)
        if self.dim_reduction is not None:
            visual_words = self.dim_reduction.transform(visual_words)

        # Predict labels
        return self.classifier.predict_proba(visual_words)

    def score(self, X, y):
        """
        Score of the classifier
        :param X: descriptors of the images
        :param y: labels of the images
        :return: score of the classifier
        """
        # Get  descriptors
        kpts, descriptors = self.detector.compute_descriptors(X, train=False)

        # Get visual words
        visual_words = self.calculate_visual_words(kpts, descriptors)

        # Get dimensionality reduction (if any)
        if self.dim_reduction is not None:
            visual_words = self.dim_reduction.transform(visual_words)

        # Compute score
        score = self.classifier.score(visual_words, y)
        print("Score:", score)
        return self.classifier.score(visual_words, y)
