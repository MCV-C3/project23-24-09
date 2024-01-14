import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import *
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input
from keras.utils import plot_model
import numpy as np
from PIL import Image
from sklearn.feature_extraction import image
import wandb

def build_mlp(input_size, hidden_layers, phase='train'):
    model = Sequential()
    model.add(Input(shape=(input_size, input_size, 3,),name='input'))
    model.add(Reshape((input_size*input_size*3,)))

    for i, el in enumerate(hidden_layers):
      if i != len(hidden_layers) - 1:
          model.add(Dense(units=el, activation='relu'))
      else:
          model.add(Dense(units=el, activation='relu', name='last'))

    if phase=='test':
        model.add(Dense(units=8, activation='linear')) # In test phase we softmax the average output over the image patches
    else:
        model.add(Dense(units=8, activation='softmax'))
    return model

def patch_train():
    """
    Train the model using the train and validation data with sweeps
    """
    wandb.init(project="noel-test")
    PATCH_SIZE = wandb.config.patch_size
    BATCH_SIZE = wandb.config.batch_size
    hidden_layers = wandb.config.hidden_layers
    epochs = 2

    DATASET_DIR = '../MIT_split'
    PATCHES_DIR = '../data/MIT_split_patches' + str(PATCH_SIZE)

    if not os.path.exists(DATASET_DIR):
      print('ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
      quit()
    if not os.path.exists(PATCHES_DIR):
      print('WARNING: patches dataset directory '+PATCHES_DIR+' does not exist!\n')
      print('Creating image patches dataset into '+PATCHES_DIR+'\n')
      generate_image_patches_db(DATASET_DIR,PATCHES_DIR,patch_size=PATCH_SIZE)
      print('patxes generated!\n')

    # Load and preprocess the training dataset
    train_dataset = keras.preprocessing.image_dataset_from_directory(
        directory=PATCHES_DIR + '/train/',
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        image_size=(PATCH_SIZE, PATCH_SIZE),
        shuffle=True,
        validation_split=None,
        subset=None
    )

    # Load and preprocess the validation dataset
    validation_dataset = keras.preprocessing.image_dataset_from_directory(
        directory=PATCHES_DIR + '/test/',
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        image_size=(PATCH_SIZE, PATCH_SIZE),
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

    model = build_mlp(PATCH_SIZE, hidden_layers)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    print(model.summary())

    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        verbose=1,
        callbacks=[wandb.keras.WandbCallback()]
    )
    MODEL_FNAME = '../current_patch_based_mlp.weights.h5'
    model.save_weights(MODEL_FNAME)


    print('Evaluating model...\n')
    test_model = build_mlp(PATCH_SIZE, hidden_layers, phase='test')
    test_model.load_weights(MODEL_FNAME)

    directory = DATASET_DIR + '/test'
    classes = {'coast': 0, 'forest': 1, 'highway': 2, 'inside_city': 3, 'mountain': 4, 'Opencountry': 5, 'street': 6,
               'tallbuilding': 7}
    correct = 0.
    total = 807
    count = 0

    for class_dir in os.listdir(directory):
        cls = classes[class_dir]
        for imname in os.listdir(os.path.join(directory, class_dir)):
            im = Image.open(os.path.join(directory, class_dir, imname))
            patches = view_as_blocks(np.array(im), block_shape=(PATCH_SIZE, PATCH_SIZE, 3)).reshape(-1, PATCH_SIZE,
                                                                                                    PATCH_SIZE, 3)
            out = test_model.predict(patches / 255.)
            predicted_cls = np.argmax(softmax(np.mean(out, axis=0)))
            if predicted_cls == cls:
                correct += 1
            count += 1
            print('Evaluated images: ' + str(count) + ' / ' + str(total), end='\r')

    print('Done!\n')
    print('Test Acc. = ' + str(correct / total) + '\n')
    wandb.log({'test_acc_aggregated': correct / total})


if __name__ == '__main__':
    wandb.login()

    print('Setting up data ...\n')

    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'patch_size': {
                'values': [128, 64, 32]
            },
            'batch_size': {
                'values': [16, 32, 64, 128]
            },
            'hidden_layers': {
                'values': [[2048, 1024, 512, 256], [2048, 1024, 512], [2048, 1024], [2048]]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="aggregated-test")
    wandb.agent(sweep_id, function=patch_train, count=25)
