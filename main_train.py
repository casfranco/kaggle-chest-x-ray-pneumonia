"""
Main script for model training
- Seed
- Load configuration hyperparameters
- Load dataset and tf.keras based generator (image processing)
- Model training
- Model evaluation and results

If executed as main script on terminal:

>> python main_train.py

"""
from casfml.models.model_trainer import ModelTrainer
from casfml.data_loader.pneumonia_kaggle import PneumoniaKaggleLoader
from casfml.utils.json_hyperparameters import Params
from casfml.utils.random import seed_everything

import warnings
import shutil
import os

warnings.filterwarnings("ignore")


def main_train():
    # seed everything (numpy, tensorflow, ...)
    seed_everything(seed=21)
    
    # Configuration file
    hparams = Params('data/configuration_files/cls_params.json')

    # Dataset handler
    pneumoniaKaggle = PneumoniaKaggleLoader(hparams)
    pneumoniaKaggle.create_data_generators()

    # Trainer object
    trainer = ModelTrainer(hparams)

    # Train
    trainer.train(pneumoniaKaggle.train_generator,
                  pneumoniaKaggle.val_generator)

    # Evaluate and compute performance metrics over the validation set
    if(hparams.evaluate_validation_metrics):
        trainer.test_model(generator=pneumoniaKaggle.val_generator)


if __name__ == "__main__":
    main_train()
