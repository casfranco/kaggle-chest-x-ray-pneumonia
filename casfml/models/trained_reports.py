from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from casfml.utils.general_plots import visualize_from_dict
from casfml.utils.json_hyperparameters import Params

import os
from os.path import join
import pandas as pd
import numpy as np
import cv2

class TrainedReports(object):
    def __init__(self) -> None:
        self.experiment_list = []
        self.base_results_path = 'data/results'
        self.history_result_folder = 'history_results'
        self.metrics_evaluation_folder = 'metrics_evaluation'
        self.trained_model_folder = 'trained_model'

    def simplify_experiment_name(self,experiment_name):
        '''Create a simpler experiment name from the trained params file
        '''
        trained_params = join(self.base_results_path,experiment_name,self.trained_model_folder,'trained_params.json')
        hparams = Params(trained_params)
        if(hparams.save_prefix != ""):
            simpler_experiment_name = f'{hparams.save_prefix}_{hparams.model}_{hparams.image_height}x{hparams.image_width}'
        else:
            simpler_experiment_name = f'{hparams.model}_{hparams.image_height}x{hparams.image_width}'

        return simpler_experiment_name

    def get_group_images_dict(self,experiment_list,image_filepath):
        experiment_images_dict = {}
        for experiment in experiment_list:
            simpler_experiment_name = self.simplify_experiment_name(experiment)
            experiment_image_path = join(self.base_results_path,experiment,image_filepath)
            experiment_image = cv2.imread(experiment_image_path)
            experiment_images_dict[simpler_experiment_name] = experiment_image

        return experiment_images_dict

    def compare_learning_curves(self,experiment_list,metric = 'accuracy'):
        image_filepath = join(self.history_result_folder,f'training_history_{metric}.png')
        experiment_images_dict = self.get_group_images_dict(experiment_list,image_filepath)  
        visualize_from_dict(experiment_images_dict)

    def compare_confusion_matrix(self,experiment_list):
        image_filepath = join(self.metrics_evaluation_folder,'validation-cm.png')
        experiment_images_dict = self.get_group_images_dict(experiment_list,image_filepath)
        visualize_from_dict(experiment_images_dict)

    def compare_metrics(self,experiment_list):
        csv_filepath = join(self.metrics_evaluation_folder,'validation_y_pred_y_true_filenames.csv')

        simpler_experiment_name = []
        f1_score_list = []
        accuracy_score_list = []
        precision_score_list = []
        recall_score_list = []

        for experiment in experiment_list:
            df_filepath = join(self.base_results_path,experiment,csv_filepath)
            df = pd.read_csv(df_filepath)
            simpler_experiment_name.append(self.simplify_experiment_name(experiment))

            y_true = df['y_true'].to_numpy()
            y_pred = df['y_pred'].to_numpy()
            accuracy_score_list.append(accuracy_score(y_true,y_pred))
            precision_score_list.append(precision_score(y_true,y_pred)) 
            recall_score_list.append(recall_score(y_true,y_pred))
            f1_score_list.append(f1_score(y_true,y_pred))

        data = {'experiment_name':simpler_experiment_name,
                'accuracy':accuracy_score_list,
                'precision':precision_score_list,
                'recall':recall_score_list,
                'f1': f1_score_list
                }

        df_compare_metrics = pd.DataFrame(data)
        print(df_compare_metrics.to_string(index=False))

