import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import wandb
import warnings
from data_processing import get_processed_data
from train import train

warnings.filterwarnings("ignore")


def training():
    wandb.init(project="t3-example")
    config = wandb.config
    train_dataset, validation_dataset, test_dataset = get_processed_data(**config)
    train(train_dataset, validation_dataset, test_dataset, **config)


if __name__ == '__main__':
    wandb.login()
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'model': {
                'values': ['convnexttiny']
            },
            'n_frozen_layers': {
                'values': [28, 53, 126]
            },
            'NUMBER_OF_EPOCHS': {
                'values': [10, 20, 40]
            },
            'optimizer': {
                'values': ['SGD', 'Adam', 'RMSprop', 'Nadam']
            },
            'lr': {
                'values': [0.1, 0.01, 0.001, 0.00005]
            },
            'momentum': {
                'values': [0.0, 0.4, 0.9]
            },
            'data_augmentation': {
                'values': [True]
            },
            'IMG_WIDTH': {
                'values': [224]
            },
            'IMG_HEIGHT': {
                'values': [224]
            },
            'BATCH_SIZE': {
                'values': [8, 16, 25, 32]
            },
            'DATASET_DIR': {
                'values': ['./MIT_small_train_1']
            },
            'dropout_before': {
                'values': ['no_dropout']
            },
            'dropout_batch_norm_after': {
                'values': [True]
            },
            'dropout': {
                'values': [0.1]
            }

        }
    }

    sweep_id = wandb.sweep(sweep_config, project="hyperparameters-tuning")
    wandb.agent(sweep_id, function=training, count=66)
