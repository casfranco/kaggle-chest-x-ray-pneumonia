import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from mlxtend.plotting import plot_confusion_matrix

# ---------------------------------------------------------------------------- #
#                                Learning curves                               #
# ---------------------------------------------------------------------------- #

def plot_all_learning_curves(csv_history_file, metrics = ['loss']):
    """Plot learning curves

    Args:
        csv_history_file (csv): info about training metrics history
    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    try:
        df = pd.read_csv(csv_history_file)
    except:
        return
    if( len(df) == 0 ):
        return
    # Images will be stored in the same directory of the csv file

    folder_path = os.path.dirname(csv_history_file)

    #Extract automatically metrics names for csv_file
    # Delete the words 'epoch' and all other works begining with 'val' or 'test'
    col_names = df.columns.values.tolist()
    char_list = ['epoch', 'val','test']
    metrics = [ele for ele in col_names if all(ch not in ele for ch in char_list)]


    for metric in metrics:
        filename = os.path.join(folder_path, 'training_history_' + metric + '.png')
        train_metric = df[metric].tolist()
        val_metric = df['val_' + metric].tolist()

        plot_learning_curve(filename, train_metric, val_metric, metric)


def plot_learning_curve(filename, train_metric, val_metric=[], metric_name=''):
    """save individual learning curves

    Args:
        filename (str): filename path
        train_metric (list): list of values for a training metric
        val_metric (list, optional): list of values for a training metric. Defaults to [].
        metric_name (str, optional): metric name. Defaults to ''.
    """
    epoch_count = list(range(1, len(train_metric) + 1))

    plt.figure()
    plt.plot(epoch_count, train_metric, 'r--')
    plt.plot(epoch_count, val_metric, 'b-')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.title('Model ' + metric_name)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.savefig(filename)
    plt.close()

# ---------------------------------------------------------------------------- #
#                            Classification metrics                            #
# ---------------------------------------------------------------------------- #

def plot_mlxtend_cm(params, cm, target_names_list = None, dir_to_save = None, cm_path = None):

    if (target_names_list == None):
        target_names_list = ['neg_class','pos_class']
    
    if ((dir_to_save is None) and (cm_path is None)):
        print(f'\n Confusion Matrix:\n{cm}')
        print('[Warning] This matrix was not save to a file. Use dir_to_save parameter to save in a specifica directory.')
        return np.array([[-1, -1 ],
                    [-1, -1]])


    class_names = target_names_list

    binary1 = cm
    fig, ax = plot_confusion_matrix(conf_mat=binary1,
                                    show_absolute=True,
                                    show_normed=True,
                                    class_names=class_names,
                                    figsize=(8, 6))

    # Saves to file
    #dir_script = os.path.dirname(dir_to_save) 
    if (cm_path is None):
        cm_filename = params.model + "_" + params.timestr + ".png"
        cm_path = os.path.join(dir_to_save, cm_filename)
    
    plt.savefig(cm_path)
    plt.close
    print(f"Confusion Matrix saved [mlxtend library]:\n{str(cm_path)}")


def plot_roc_auc(fpr,tpr,auc_roc,metrics_evaluation_path,dataset):
    #plt.figure(1, figsize=(10, 10))
    plot_filename = os.path.join(metrics_evaluation_path, dataset + '_roc_curves.png')
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')

    # plot roc curves
    plt.plot(fpr, tpr, linestyle='--',color='orange', label=dataset + " (AUC: " + str(round(auc_roc, 3)) + ")")

    # title
    plt.title(dataset + ' ROC curve')
    # x label
    plt.xlabel('False Positive Rate')
    # y label
    plt.ylabel('True Positive rate')

    plt.legend(loc='best')
    plt.savefig(plot_filename) #,dpi=300)
    plt.close