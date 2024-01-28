import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import wandb
import warnings
from t4 import train
from data_processing import get_processed_data

warnings.filterwarnings("ignore")


def training():
    wandb.init(project="t4-example")
    config = wandb.config
    train_dataset, validation_dataset, test_dataset = get_processed_data(**config)
    train(train_dataset, validation_dataset, test_dataset, **config)


if __name__ == '__main__':
    wandb.login()
    sweep_config = {
        'name': 'random-search',
        'method': 'random',
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'NUMBER_OF_EPOCHS': {
                'values': [50]
            },
            'optimizer': {
                'values': ['Adam', 'SGD', 'Nadam']
            },
            'lr': {
                'values': [0.0001, 0.001, 0.01]
            },
            'momentum': {
                'values': [0.9]
            },
            'IMG_WIDTH': {
                'values': [32, 64, 128, 224]
            },
            'IMG_HEIGHT': {
                'values': [224]
            },
            'BATCH_SIZE': {
                'values': [8, 16, 32, 64]
            },
            'DATASET_DIR': {
                'values': ['./MIT_small_train_1_augmented']
            },
            'conv2d': {
                'values': [
                    {
                        'n_convs': [2, 2, 3],
                        'n_filters': [8, 16, 32],
                        'pooling': [[2, 'max'], [2, 'max'], [2, 'max']],
                        'padding': 'same'
                    },
                    {
                        'n_convs': [1, 2],
                        'n_filters': [32, 64],
                        'pooling': [[2, 'max'], [2, 'max']],
                        'padding': 'same'
                    },
                    {
                        'n_convs': [2, 2, 2],
                        'n_filters': [16, 16, 32],
                        'pooling': [[2, 'max'], [2, 'max'], [2, 'max']],
                        'padding': 'same'
                    }
                ]
            },
            'normalization_dense': {
                'values': [None, 'batch']
            },
            'n_units': {
                'values': [None, 32, 64, 128, 256, 512]
            },
            'dropout': {
                'values': [0.5]
            },
            'kernel_regularizer': {
                'values': [
                    None
                    #['l1', 0.01]
                    # ['l2', 0.01]
                    # ['l1_l2', 0.01],
                    # ['orthogonal', 0.01]
                ]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="random-search")
    wandb.agent(sweep_id, function=training, count=50)
