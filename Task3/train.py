import keras.callbacks

import wandb
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16
from keras.applications.convnext import ConvNeXtTiny
from keras.utils import plot_model
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam


def build_optimizer(optimizer, lr, momentum):
    if optimizer == 'SGD':
        opt = SGD(learning_rate=lr, momentum=momentum)
    elif optimizer == 'Adam':
        opt = Adam(learning_rate=lr)
    elif optimizer == 'RMSprop':
        opt = RMSprop(learning_rate=lr, momentum=momentum)
    elif optimizer == 'Adagrad':
        opt = Adagrad(learning_rate=lr)
    elif optimizer == 'Adadelta':
        opt = Adadelta(learning_rate=lr)
    elif optimizer == 'Adamax':
        opt = Adamax(learning_rate=lr)
    elif optimizer == 'Nadam':
        opt = Nadam(learning_rate=lr)
    else:
        raise ValueError('Unknown optimizer: ' + optimizer)
    return opt


def train(train_dataset, validation_dataset, test_dataset, **kwargs):
    # create the base pre-trained model
    chosen_model = kwargs['model']
    N = kwargs['n_frozen_layers']
    NUMBER_OF_EPOCHS = kwargs['NUMBER_OF_EPOCHS']
    callbacks = kwargs.get('callbacks', [])
    optimizer_str = kwargs['optimizer']
    lr = kwargs['lr']
    momentum = kwargs['momentum']
    dropout_BEFORE = kwargs['dropout_before']
    dropout_batch_norm_AFTER = kwargs['dropout_batch_norm_after']
    dropout = kwargs['dropout']

    if chosen_model == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False)
    elif chosen_model == 'convnexttiny':
        base_model = ConvNeXtTiny(weights='imagenet', include_top=False)
    else:
        raise ValueError('Unknown model: ' + chosen_model)

    # add a global spatial average pooling layer
    x = base_model.output
    if dropout_BEFORE == 'before':
        x = keras.layers.Dropout(dropout)(x)
    x = GlobalAveragePooling2D()(x)
    if dropout_BEFORE == 'after':
        x = keras.layers.Dropout(dropout)(x)

    # let's add a fully-connected layer
    x = Dense(1024, activation=None)(x)
    if dropout_batch_norm_AFTER:
        x = keras.layers.BatchNormalization()(x)
        # Apply activation.
        x = keras.activations.relu(x)
    else:
        x = keras.activations.relu(x)
        x = keras.layers.Dropout(dropout)(x)
    predictions = Dense(8, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional VGG16 layers
    for layer in base_model.layers:
        layer.trainable = False

    plot_model(model, to_file=f'model_{chosen_model}.png', show_shapes=True, show_layer_names=True)

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # train the model on the new data for a few epochs
    history = model.fit(train_dataset,
                        epochs=20,
                        validation_data=validation_dataset,
                        verbose=1)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    # for i, layer in enumerate(base_model.layers):
    #     print(i, layer.name)

    # we chose to train the top N layers, i.e. we will freeze
    # the first layers and unfreeze the N rest:
    for layer in model.layers[:N]:
        layer.trainable = False
    for layer in model.layers[N:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    opt = build_optimizer(optimizer_str, lr, momentum)

    model.compile(optimizer=opt, metrics=['accuracy'],
                  loss='categorical_crossentropy')

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    history = model.fit(train_dataset,
                        epochs=NUMBER_OF_EPOCHS,
                        validation_data=validation_dataset,
                        verbose=1,
                        callbacks=[wandb.keras.WandbCallback(),
                                   wandb.keras.WandbModelCheckpoint(f"{chosen_model}_best_model", monitor="val_accuracy",
                                                                    save_best_only=True)])

    result = model.evaluate(test_dataset)
    print(result)
    print(history.history.keys())
