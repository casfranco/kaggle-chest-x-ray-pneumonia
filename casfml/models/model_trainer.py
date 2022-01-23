from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.utils import class_weight

from casfml.utils.plot_metrics import plot_all_learning_curves, plot_mlxtend_cm, plot_roc_auc
from casfml.utils.file_utils import safe_make_folder, safe_make_list_folders
from casfml.utils.training_utils import PlotLearning, PlotLosses
from casfml.utils.training_utils import allow_growth
from casfml.models.model_manager import ModelManager
import time
from os.path import join
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import History, EarlyStopping, ModelCheckpoint, CSVLogger
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ModelTrainer(object):
    """
    Class responsable for create, train and evaluate the neural network model.
    Use hparams file for configuration
    Save the results > data/results/[experiment_name] 
    """
    def __init__(self, hparams):
        self.hparams = hparams
        self.model = self.create_model_architecture()
        self.define_namespace()
        self.callbacks_list = []

    def load_trained_model(self,experiment_name):
        # Update the namespace and load the best trained h5 model
        self.define_namespace(saved_experiment_name=experiment_name)
        self.model = ModelManager.load_model(self.hparams,self.h5_best_model_filepath)
    
    def define_namespace(self,saved_experiment_name=None):
        if(saved_experiment_name is not None):
            self.experiment_name = saved_experiment_name

        elif(self.hparams.save_prefix != ""):
            self.experiment_name = self.hparams.save_prefix + \
                self.hparams.model + time.strftime("_%Y_%m%d_%H%M")
        else:
            self.experiment_name = self.hparams.model + \
                time.strftime("_%Y_%m%d_%H%M")

        if(self.hparams.save_model_results):
            # Folders to save the results
            self.base_results_path = join('data/results', self.experiment_name)
            self.h5_best_model_path = join(
                self.base_results_path, "trained_model")
            self.metrics_evaluation_path = join(
                self.base_results_path, "metrics_evaluation")
            self.history_training_path = join(
                self.base_results_path, "history_results")
            
            # Filename with full path
            self.h5_best_model_filepath = join(
                self.h5_best_model_path, 'best_model.h5')
            self.history_training_csv_filepath = join(
                self.history_training_path, 'training_history_log.csv')

    def set_hparams(self, hparams):
        self.hparams = hparams

    def create_model_architecture(self):
        model = ModelManager.create_model(self.hparams)
        return model

    def compile_model(self):
        self.model.compile(loss=self.hparams.loss,
                           optimizer=self.hparams.optimizer,
                           metrics=['accuracy'])

    def create_callbacks(self, show_learning_curves=False):
        early_stop = EarlyStopping(monitor='val_loss', patience=10)

        if(self.hparams.save_model_results):
            model_checkpoint = ModelCheckpoint(
                self.h5_best_model_filepath, save_weights_only=True, save_best_only=True, mode='auto')
            csv_logger = CSVLogger(self.history_training_csv_filepath)

            self.callbacks_list = [early_stop, model_checkpoint, csv_logger]
        else:
            self.callbacks_list = [early_stop]

        if(show_learning_curves):
            plot_losses = PlotLearning()
            self.callbacks_list.append(plot_losses)

    def train(self, train_generator, val_generator, show_learning_curves=False):
        
        allow_growth(tf_version=2)
        # Create folder for save the model/results
        if(self.hparams.save_model_results):
            safe_make_list_folders([self.base_results_path, self.h5_best_model_path,
                                    self.metrics_evaluation_path, self.history_training_path])

        self.create_callbacks(show_learning_curves)
        self.compile_model()

        # 
        
        # Model Training
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=self.hparams.num_epochs,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            callbacks=self.callbacks_list,
            verbose=1
        )

        if(self.hparams.save_model_results):
            # Save the learning curves and hparms used for training
            plot_all_learning_curves(self.history_training_csv_filepath)
            self.hparams.save(
                join(self.h5_best_model_path, 'trained_params.json'))

    def test_model(self, model=None, generator=None, threshold=0.5, dataset='validation'):
        # Load Model if is None or there is a best trained model
        if((model is None) or self.hparams.save_model_results):
            print('loading best trained model')
            model = ModelManager.load_model(
                self.hparams, self.h5_best_model_filepath)

        if (generator is None):
            return

        y_true = np.array(generator.labels)
        y_score = model.predict(generator)
        y_pred = (y_score > threshold).astype(int)

        # Metrics per Label
        test_accuracy = accuracy_score(y_true, y_pred)
        test_precission = precision_score(y_true, y_pred)
        test_recall = recall_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred)

        # Print the evaluation results
        print(
            f"\n======= Metrics: {dataset} - experiment: {self.experiment_name}  =======\n")
        print(f'{dataset} Accuracy:', test_accuracy)
        print(f'{dataset} Precision:', test_precission)
        print(f'{dataset} Recall:', test_recall)
        print(f'{dataset} F1:', test_f1)

        print(f"\n======= Precision and Recall: {dataset}  =======\n")
        print(classification_report(y_true, y_pred,
                                    digits=3, target_names=self.hparams.classes))

        print(f"\n======= Confusion Matrix: {dataset}  =======\n")
        # custom function to save confusion matrix
        test_cm = confusion_matrix(y_true, y_pred)

        # ================ ROC-AUC =======================

        # RoC score
        # calculate roc curve for model
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_pred)

        # Save the results
        if(self.hparams.save_model_results):
            cm_path = join(self.metrics_evaluation_path, f'{dataset}-cm.png')

            plot_mlxtend_cm(params=self.hparams, 
                            cm=test_cm, 
                            target_names_list=self.hparams.classes, 
                            cm_path=cm_path)

            plot_roc_auc(fpr, tpr, auc_roc,
                         self.metrics_evaluation_path, dataset)

            # Save prediction data
            # Saves csv file with filenames, preds and labels
            columns = ["y_pred"]
            results = pd.DataFrame(y_pred, columns=columns)
            results["Filenames"] = generator.filenames
            ordered_cols = ["Filenames"]+columns
            results = results[ordered_cols]  # To get the same column order
            results = results.assign(y_true=y_true)
            results = results.assign(y_score_skinfold=y_score)

            results.to_csv(join(self.metrics_evaluation_path,
                                f'{dataset}_y_pred_y_true_filenames.csv'), index=False)
