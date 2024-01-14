import keras
import tensorflow as tf
from sklearn.svm import SVC
import numpy as np


def train_svm(model, train_dataset, train_labels):
    """
    Train the SVM using the features extracted from the model
    :param model: the model used to extract the features
    :param train_dataset: the dataset used to train the SVM
    :return: the trained SVM
    """
    # Extract the features from the model
    features = model.predict(train_dataset)

    # Train the SVM
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(features, train_labels)
    return svm


def test_svm(svm, model, test_dataset, test_labels):
    """
    Test the SVM using the features extracted from the model
    :param svm: the trained SVM
    :param model: the model used to extract the features
    :param test_dataset: the dataset used to test the SVM
    :return: the accuracy of the SVM
    """
    # Extract the features from the model
    features = model.predict(test_dataset)
    # Test the SVM
    accuracy = svm.score(features, test_labels)
    return accuracy


if __name__ == '__main__':
    DATASET_DIR = '../MIT_split'
    IMG_SIZE = 32
    BATCH_SIZE = 16

    # Load and preprocess the training dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        directory=DATASET_DIR + '/train/',
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=False,
        subset=None
    )

    class_names = train_dataset.class_names

    # Load and preprocess the validation dataset
    validation_dataset = keras.preprocessing.image_dataset_from_directory(
        directory=DATASET_DIR + '/test/',
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        image_size=(IMG_SIZE, IMG_SIZE),
        validation_split=None,
        shuffle=False,
        subset=None
    )

    # Data augmentation and preprocessing
    preprocessing_train = keras.Sequential([
        keras.layers.experimental.preprocessing.Rescaling(1. / 255),
        keras.layers.experimental.preprocessing.RandomFlip("horizontal")
    ])

    preprocessing_validation = keras.Sequential([
        keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    ])

    train_dataset = train_dataset.map(lambda x, y: (preprocessing_train(x, training=True), y))
    validation_dataset = validation_dataset.map(lambda x, y: (preprocessing_validation(x, training=False), y))

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset), seed=42, reshuffle_each_iteration=False)
    validation_dataset = validation_dataset.shuffle(buffer_size=len(validation_dataset), seed=42, reshuffle_each_iteration=False)

    train_labels = []
    for _, labels in train_dataset:
        for label in labels:
            array = label.numpy().argmax(axis=0)
            train_labels.append(class_names[array])
    train_labels = np.array(train_labels)

    validation_labels = []
    for _, labels in validation_dataset:
        for label in labels:
            array = label.numpy().argmax(axis=0)
            validation_labels.append(class_names[array])
    validation_labels = np.array(validation_labels)
    model = keras.models.load_model('model-best.h5')

    # accuracy = model.evaluate(validation_dataset)
    layer = 'last'
    model_layer = keras.Model(inputs=model.input, outputs=model.get_layer(layer).output)
    model_layer.summary()

    fitted_svm = train_svm(model_layer, train_dataset, train_labels)
    train_accuracy = test_svm(fitted_svm, model_layer, train_dataset, train_labels)
    test_accuracy = test_svm(fitted_svm, model_layer, validation_dataset, validation_labels)
    print('Train accuracy: ', train_accuracy)
    print('Test accuracy: ', test_accuracy)
