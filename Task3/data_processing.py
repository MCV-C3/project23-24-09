import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    data_augmentation = kwargs['data_augmentation']
    IMG_WIDTH = kwargs['IMG_WIDTH']
    IMG_HEIGHT = kwargs['IMG_HEIGHT']
    BATCH_SIZE = kwargs['BATCH_SIZE']
    DATASET_DIR = kwargs['DATASET_DIR']
    model = kwargs['model']

    if model == 'vgg16':
        preprocess_input = tf.keras.applications.vgg16.preprocess_input
    elif model == 'convnexttiny':
        preprocess_input = tf.keras.applications.convnext.preprocess_input
    else:
        raise ValueError('Unknown model: ' + model)

    if data_augmentation:
        train_data_generator = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False
        )
    else:
        train_data_generator = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

    # Load and preprocess the training dataset
    train_dataset = train_data_generator.flow_from_directory(
        directory=DATASET_DIR + '/train/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    # Load and preprocess the validation dataset
    validation_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    validation_dataset = validation_data_generator.flow_from_directory(
        directory=DATASET_DIR + '/test/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    # Load and preprocess the test dataset
    test_data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    test_dataset = test_data_generator.flow_from_directory(
        directory=DATASET_DIR + '/test/',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    return train_dataset, validation_dataset, test_dataset
