import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import keras


def get_processed_data(**kwargs):
    """
    Define the data generator for data augmentation and preprocessing
    :param kwargs: Keyword arguments
    :key: IMG_WIDTH: Image width
    :key: IMG_HEIGHT: Image height
    :key: BATCH_SIZE: Batch size
    :key: DATASET_DIR: Dataset directory
    :key: preprocess_input: Preprocessing function (e.g. from keras.applications.resnet50 import preprocess_input)
    :key: data_augmentation: Boolean flag to enable data augmentation
    :return: Train, validation and test datasets
    """
    #data_augmentation = kwargs['data_augmentation']
    IMG_WIDTH = kwargs['IMG_WIDTH']
    IMG_HEIGHT = kwargs['IMG_WIDTH']
    BATCH_SIZE = kwargs['BATCH_SIZE']
    DATASET_DIR = kwargs['DATASET_DIR']


    # Load and preprocess the training dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        directory=DATASET_DIR + '/train/',
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        shuffle=True,
        validation_split=None,
        subset=None
    )

    # Load and preprocess the validation dataset
    validation_dataset = keras.preprocessing.image_dataset_from_directory(
        directory=DATASET_DIR + '/test/',
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        image_size=(IMG_WIDTH, IMG_HEIGHT),
        shuffle=True,
        seed=123,
        validation_split=None,
        subset=None
    )

    # Data augmentation and preprocessing
    preprocessing_train = keras.Sequential([
        keras.layers.experimental.preprocessing.Rescaling(1. / 255)
        # keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        # keras.layers.experimental.preprocessing.RandomRotation(0.2),
        # keras.layers.experimental.preprocessing.RandomZoom(0.2),
        # keras.layers.experimental.preprocessing.RandomContrast(0.2)
    ])

    preprocessing_validation = keras.Sequential([
        keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    ])

    preprocessing_test = keras.Sequential([
        keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    ])

    train_dataset = train_dataset.map(lambda x, y: (preprocessing_train(x, training=True), y))
    validation_dataset = validation_dataset.map(lambda x, y: (preprocessing_validation(x, training=False), y))
    test_dataset = validation_dataset.map(lambda x, y: (preprocessing_test(x, training=False), y))

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset


