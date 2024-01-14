import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

import matplotlib
matplotlib.use('Agg')

import wandb
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input


DATASET_DIR = '../MIT_split'


def train():
    """
    Train the model using the train and validation data with sweeps
    """
    wandb.init(project="fran-test")
    IMG_SIZE = wandb.config.img_size
    BATCH_SIZE = wandb.config.batch_size
    hidden_layers = wandb.config.hidden_layers
    epochs = 50

    # Load and preprocess the training dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        directory=DATASET_DIR + '/train/',
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        image_size=(IMG_SIZE, IMG_SIZE),
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
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=True,
        seed=123,
        validation_split=None,
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

    model = Sequential()
    input = Input(shape=(IMG_SIZE, IMG_SIZE, 3,), name='input')
    model.add(input)  # Input tensor
    model.add(Reshape((IMG_SIZE * IMG_SIZE * 3,), name='reshape'))

    for i, el in enumerate(hidden_layers):
        if i != len(hidden_layers) - 1:
            model.add(Dense(units=el, activation='relu'))
        else:
            model.add(Dense(units=el, activation='relu', name='last'))

    model.add(Dense(units=8, activation='softmax', name='classification'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        verbose=1,
        callbacks=[wandb.keras.WandbCallback()]
    )


wandb.login()
print('Setting up data ...\n')

sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'img_size': {
            'values': [256, 128, 64, 32]
        },
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'hidden_layers': {
            'values': [[2048, 1024, 512, 256], [2048, 1024, 512], [2048, 1024], [2048]]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="fran-test")
wandb.agent(sweep_id, function=train, count=25)

