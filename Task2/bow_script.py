import keras
import tensorflow as tf
from sklearn.svm import SVC
from utils import *
from sklearn.cluster import MiniBatchKMeans
import wandb


def get_visual_words(codebook, descriptor):
    """
    Visual words
    :param descriptor: descriptor
    :param k: number of clusters
    :return: visual words
    """
    if descriptor is not None and descriptor.shape[0] > 0:
        words = codebook.predict(descriptor)
        return np.bincount(words, minlength=codebook.n_clusters)
    return np.zeros(codebook.n_clusters)


def compute_visual_words(codebook, descriptors):
    """
    Compute the visual words of a list of descriptors
    :param descriptors: descriptors of the images
    :return: visual words
    """
    if type(descriptors) is list:
        visual_words=np.zeros((len(descriptors), codebook.n_clusters),dtype=np.float32)
        for i in range(len(descriptors)):
            visual_words[i,:] = get_visual_words(codebook, descriptors[i])
        return visual_words
    elif type(descriptors) is np.ndarray and descriptors.ndim == 2 and descriptors.shape[0] == 1:
        return get_visual_words(codebook, descriptors)
    else:
        print("Wrong type of descriptors, must be list or numpy array")
        return None


def train_bow(model, train_dataset, train_labels, batch_size, k=256):
    """
    Train the SVM using the features extracted from the model
    :param model: the model used to extract the features
    :param train_dataset: the dataset used to train the SVM
    :return: the trained SVM
    """
    # Extract the features from the model
    train_descriptors = []
    print("Getting descriptors...")
    prediction_array = model.predict(train_dataset)
    for i in range(0, len(prediction_array), batch_size):
        train_descriptors.append(prediction_array[i:i+batch_size])

    print("Computing codebook...")
    codebook = MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20, compute_labels=False, reassignment_ratio=10**-4, random_state=42)
    codebook.fit(np.vstack(train_descriptors))

    print("Computing visual words...")
    visual_words = compute_visual_words(codebook, train_descriptors)

    new_train_labels = train_labels[::batch_size]

    # Train the SVM
    print("Training SVM...")
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(visual_words, new_train_labels)
    print("Finished training SVM!")
    return codebook, svm


def test_bow(svm, codebook, model, test_dataset, test_labels, batch_size):
    """
    Test the SVM using the features extracted from the model
    :param svm: the trained SVM
    :param model: the model used to extract the features
    :param test_dataset: the dataset used to test the SVM
    :return: the accuracy of the SVM
    """
    # Extract the features from the model
    test_descriptors = []
    prediction_array = model.predict(test_dataset)
    print("Getting descriptors from test...")
    for i in range(0, len(prediction_array), batch_size):
        test_descriptors.append(prediction_array[i:i+batch_size])

    print("Computing visual words from test...")
    visual_words = compute_visual_words(codebook, test_descriptors)

    # Test the SVM
    print("Testing SVM...")
    new_test_labels = test_labels[::batch_size]
    accuracy = svm.score(visual_words, new_test_labels)
    return accuracy


if __name__ == '__main__':
    # user defined variables
    PATCH_SIZE = 32
    BATCH_SIZE = int(256 / PATCH_SIZE)**2
    DATASET_DIR = '../MIT_split'
    PATCHES_DIR = '../data/MIT_split_patches' + str(PATCH_SIZE)
    MODEL_FNAME = '../4th-patch-model-best.h5'
    ks = [64, 128, 256, 512]

    wandb.login()
    wandb.init(project="bow", name="4th-best")

    if not os.path.exists(DATASET_DIR):
        print('ERROR: dataset directory ' + DATASET_DIR + ' do not exists!\n')
        quit()
    if not os.path.exists(PATCHES_DIR):
        print('WARNING: patches dataset directory ' + PATCHES_DIR + ' does not exist!\n')
        print('Creating image patches dataset into ' + PATCHES_DIR + '\n')
        generate_image_patches_db(DATASET_DIR, PATCHES_DIR, patch_size=PATCH_SIZE)
        print('patxes generated!\n')

    # Data augmentation and preprocessing
    preprocessing_train = keras.Sequential([
        keras.layers.experimental.preprocessing.Rescaling(1. / 255),
        keras.layers.experimental.preprocessing.RandomFlip("horizontal")
    ])

    preprocessing_validation = keras.Sequential([
        keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    ])

    # Load and preprocess the training dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        directory=PATCHES_DIR + '/train/',
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        image_size=(PATCH_SIZE, PATCH_SIZE),
        shuffle=False,
        subset=None
    )

    class_names = train_dataset.class_names

    # Load and preprocess the validation dataset
    validation_dataset = keras.preprocessing.image_dataset_from_directory(
        directory=PATCHES_DIR + '/test/',
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        image_size=(PATCH_SIZE, PATCH_SIZE),
        shuffle=False,
        subset=None
    )

    train_dataset = train_dataset.map(lambda x, y: (preprocessing_train(x, training=True), y))
    validation_dataset = validation_dataset.map(lambda x, y: (preprocessing_validation(x, training=False), y))

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

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
    model = keras.models.load_model(MODEL_FNAME)

    # accuracy = model.evaluate(validation_dataset)
    layer = 'last'
    model_layer = keras.Model(inputs=model.input, outputs=model.get_layer(layer).output)
    model_layer.summary()
    wandb.define_metric(f"k")
    wandb.define_metric(f"train_accuracy", step_metric=f"k")
    wandb.define_metric(f"test_accuracy", step_metric=f"k")
    for k in ks:
        print(f"Training for k={k}")
        fitted_codebook, fitted_svm = train_bow(model_layer, train_dataset, train_labels, BATCH_SIZE, k)
        train_accuracy = test_bow(fitted_svm, fitted_codebook, model_layer, train_dataset, train_labels, BATCH_SIZE)
        test_accuracy = test_bow(fitted_svm, fitted_codebook, model_layer, validation_dataset, validation_labels, BATCH_SIZE)
        print('Train accuracy: ', train_accuracy)
        print('Test accuracy: ', test_accuracy)
        log_dict = {f'k': k, f'train_accuracy': train_accuracy, f'test_accuracy': test_accuracy}
        wandb.log(log_dict)
