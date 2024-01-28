import keras.callbacks

import wandb
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, InputLayer, Dropout, \
    BatchNormalization, Activation, AveragePooling2D, GroupNormalization
from keras.applications.vgg16 import VGG16
from keras.applications.convnext import ConvNeXtTiny
from keras.utils import plot_model
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from data_processing import get_processed_data


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

def get_kernel_regularizer(kernel_regularizer, factor):
    if kernel_regularizer == 'l1':
        return keras.regularizers.l1(factor)
    elif kernel_regularizer == 'l2':
        return keras.regularizers.l2(factor)
    elif kernel_regularizer == 'l1_l2':
        return keras.regularizers.l1_l2(l1=factor, l2=factor)
    elif kernel_regularizer == 'orthogonal':
        return keras.regularizers.OrthogonalRegularizer(factor)
    else:
        return None


def train(train_dataset, validation_dataset, test_dataset, **kwargs):
    # create the base pre-trained model
    NUMBER_OF_EPOCHS = kwargs['NUMBER_OF_EPOCHS']
    callbacks = kwargs.get('callbacks', [])
    optimizer_str = kwargs['optimizer']
    lr = kwargs['lr']
    momentum = kwargs['momentum']
    IMG_WIDTH = kwargs['IMG_WIDTH']
    IMG_HEIGHT = kwargs['IMG_WIDTH']
    conv2d = kwargs['conv2d']
    n_convs = conv2d['n_convs']
    n_filters = conv2d['n_filters']
    pooling = conv2d['pooling']
    normalization_dense = kwargs['normalization_dense']
    n_units = kwargs['n_units']
    dropout = kwargs['dropout']
    batch_size = kwargs['BATCH_SIZE']
    kernel_regularizer = kwargs['kernel_regularizer']
    padding = conv2d['padding']

    k_r = None
    if kernel_regularizer is not None:
        k_r = get_kernel_regularizer(kernel_regularizer[0], kernel_regularizer[1])


    # create model
    model = Sequential()
    model.add(InputLayer(input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), batch_size=batch_size))
    for i, (n, f, p, pad) in enumerate(zip(n_convs, n_filters, pooling, padding)):
        for j in range(n):
            model.add(Conv2D(f, (3, 3), activation='relu', padding='same', kernel_regularizer=k_r))
                             # kernel_initializer='glorot_normal', bias_initializer='glorot_normal'))
        if p[-1] == 'max':
            model.add(MaxPooling2D((p[0], p[0]), strides=2, padding='valid'))
        elif p[-1] == 'avg':
            model.add(AveragePooling2D((p[0], p[0])))

    model.add(Flatten())

    if n_units is not None:
        if normalization_dense is not None:
            model.add(Dense(n_units, activation=None))# , kernel_regularizer=k_r))
            if normalization_dense == 'batch':
                model.add(BatchNormalization())
            elif normalization_dense == 'group':
                model.add(GroupNormalization())
            model.add(Activation('relu'))
        else:
            model.add(Dense(n_units, activation='relu'))# , kernel_regularizer=k_r))

        if dropout is not None:
            model.add(Dropout(dropout))

    model.add(Dense(8, activation='softmax'))

    model.summary()

    # compile the model (should be done *after* setting layers to non-trainable)
    optimizer = build_optimizer(optimizer_str, lr, momentum)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # train the model on the new data for a few epochs
    history = model.fit(train_dataset,
                        epochs=NUMBER_OF_EPOCHS,
                        validation_data=validation_dataset,
                        verbose=1,
                        callbacks=[wandb.keras.WandbCallback()])

    result = model.evaluate(test_dataset)
    print(result)
