import shutil
import os
from os.path import join
from matplotlib.pyplot import show
import numpy as np
import pandas as pd

from casfml.data_loader.data_loader_base import DataLoader 
from casfml.data_loader.keras_generators import KerasClsGenerators
from casfml.utils.file_utils import get_filenames

class PneumoniaKaggleLoader(DataLoader):
    
    def __init__(self,hparams):
        self.data_path = 'data/datasets/chest_xray'
        self.df_train, self.df_validation = self.__create_train_validation_dataframes(0.25)
        self.df_test = self.__create_test_dataframes()
        self.hparams = hparams

    def download_data(self):
        pass
        
    def create_split_training_with_balanced_validation_data(self,normal_percentage_data_from_training):
        self.df_train, self.df_validation = self.__create_train_validation_dataframes(normal_percentage_data_from_training)
    
    def __create_train_validation_dataframes(self,normal_percentage_data_from_training):
        np.random.seed(seed=21)
        path_train_normal = join(self.data_path,'train/NORMAL')
        path_train_pneumonia = join(self.data_path,'train/PNEUMONIA')
       
        files_train_normal = [os.path.join(path_train_normal,i) for i in os.listdir(path_train_normal) if os.path.isfile(os.path.join(path_train_normal,i))]
        files_train_pneumonia = [os.path.join(path_train_pneumonia,i) for i in os.listdir(path_train_pneumonia) if os.path.isfile(os.path.join(path_train_pneumonia,i))]

        # index para escolher o n% de amostras normais
        random_index = np.random.permutation(len(files_train_normal))
        validation_samples = np.floor(len(random_index)*normal_percentage_data_from_training).astype(int)

        training_idx, validation_idx = random_index[validation_samples:], random_index[:validation_samples]
        training_paths_normal_selected = [files_train_normal[i] for i in training_idx]
        training_labels_normal = [0] * len(training_paths_normal_selected)

        validation_paths_normal_selected = [files_train_normal[i] for i in validation_idx]
        validation_labels_normal = [0] * len(validation_paths_normal_selected)

        random_index_pneumonia = np.random.permutation(len(files_train_pneumonia))

        pneumonia_training_idx, pneumonia_validation_idx = random_index_pneumonia[validation_samples:], random_index[:validation_samples]
        training_paths_pneumonia_selected = [files_train_pneumonia[i] for i in pneumonia_training_idx]
        training_labels_pneumonia = [1]*len(training_paths_pneumonia_selected)

        validation_paths_pneumonia_selected = [files_train_pneumonia[i] for i in pneumonia_validation_idx]
        validation_labels_pneumonia = [1]*len(validation_paths_pneumonia_selected)

        training_paths_selected = training_paths_normal_selected + training_paths_pneumonia_selected
        training_labels = training_labels_normal + training_labels_pneumonia

        validation_paths_selected = validation_paths_normal_selected + validation_paths_pneumonia_selected
        validation_labels = validation_labels_normal + validation_labels_pneumonia

        df_train = pd.DataFrame(list(zip(training_paths_selected,training_labels)),
                        columns = ['filepath','label'])

        df_train = df_train.sample(frac=1).reset_index(drop=True)

        df_validation = pd.DataFrame(list(zip(validation_paths_selected,validation_labels)),
                        columns = ['filepath','label'])

        df_validation = df_validation.sample(frac=1).reset_index(drop=True)
        
        # labels should be strings to work with keras generator
        df_train['label'] = df_train['label'].astype(str)
        df_validation['label'] = df_validation['label'].astype(str)

        return df_train, df_validation

    def __create_test_dataframes(self):
        path_test_normal = join(self.data_path,'test/NORMAL')
        path_test_pneumonia = join(self.data_path,'test/PNEUMONIA')

        files_test_normal = [os.path.join(path_test_normal,i) for i in os.listdir(path_test_normal) if os.path.isfile(os.path.join(path_test_normal,i))]
        files_test_pneumonia = [os.path.join(path_test_pneumonia,i) for i in os.listdir(path_test_pneumonia) if os.path.isfile(os.path.join(path_test_pneumonia,i))]

        labels_test_normal = [0]*len(files_test_normal)
        labels_test_pneumonia = [1]*len(files_test_pneumonia)

        test_files = files_test_normal + files_test_pneumonia
        test_labels = labels_test_normal + labels_test_pneumonia
    
        df = pd.DataFrame(list(zip(test_files,test_labels)),
                    columns = ['filepath','label'])

        df = df.sample(frac=1).reset_index(drop=True)
        # labels should be strings to work with keras generator
        df['label'] = df['label'].astype(str)
        return df 

    def create_data_generators(self,augmentations=False):
        image_size = (self.hparams.image_height,self.hparams.image_width)
        
        # Train generator only that should apply data augmentations
        if(self.hparams.apply_data_augmentation):
            self.train_generator = KerasClsGenerators.get_generator(self.hparams.batch_size, image_size,self.df_train,self.hparams.data_augmentation,shuffle=True)
        else:
            self.train_generator = KerasClsGenerators.get_generator(self.hparams.batch_size, image_size,self.df_train,data_augmentation=None,shuffle=True)

        # Validation generator
        self.val_generator = KerasClsGenerators.get_generator(1, image_size,self.df_validation,data_augmentation=None,shuffle=False)

        # Test generator
        self.test_generator = KerasClsGenerators.get_generator(1, image_size,self.df_test,data_augmentation=None,shuffle=False)

    def get_single_generator(self,dataset='validation',augmentations=False,shuffle=False):
        image_size = (self.hparams.image_height,self.hparams.image_width)
        df_name = 'df_'+ dataset    

        df = getattr(self,df_name)
        generator = KerasClsGenerators.get_generator(1, image_size,df,data_augmentation=None,suffle=shuffle)
        return generator

    def plot_original_data_distribution(self):
        ''' Display the original dataset distribution based on the number of images per folder
        '''


        datasets = ['train','val','test']
        normal_distribution = []
        pneumonia_distribution = []
        for dataset in datasets:
            normal_distribution.append(len(get_filenames(join(self.data_path,dataset,'NORMAL'))))

        for dataset in datasets:
            pneumonia_distribution.append(len(get_filenames(join(self.data_path,dataset,'PNEUMONIA'))))
        
        self.plot_data_distribution(datasets,normal_distribution,pneumonia_distribution)

    def plot_current_training_data_distribution(self):
        ''' Display the current dataset distrution, used for training.
        '''
        datasets = ['train','val','test']
        normal_distribution = [(self.df_train['label']=='0').sum(),(self.df_validation['label']=='0').sum(),(self.df_test['label']=='0').sum()]
        pneumonia_distribution = [(self.df_train['label']=='1').sum(),(self.df_validation['label']=='1').sum(),(self.df_test['label']=='1').sum()]
        self.plot_data_distribution(datasets,normal_distribution,pneumonia_distribution)


    def plot_data_distribution(self,datasets,normal_distribution,pneumonia_distribution):
        import matplotlib.pyplot as plt
        x = np.arange(len(datasets))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, normal_distribution, width, label='Normal')
        rects2 = ax.bar(x + width/2, pneumonia_distribution, width, label='Pneumonia')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Numero de imagens')
        ax.set_title('Numero de imagens por dataset e label')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()


        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()

        plt.show()

    def show_samples(self,label):
        if(label=='normal'):
            label_code = '0'
        else:
            label_code = '1'

        samples_normal = self.df_validation[self.df_validation['label']==label_code]['filepath'].to_list()
        samples_normal = samples_normal[:6]
        self.show_images(samples_normal)

    def show_images(self,img_paths, rows=2, cols=3):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import cv2

        assert len(img_paths) <= rows*cols 
        
        mpl.rc('font', size=8)
        plt.figure(figsize=(15, 8)) 
        grid = gridspec.GridSpec(rows, cols) 

        for idx, img_path in enumerate(img_paths):
            image = cv2.imread(img_path) 
            ax = plt.subplot(grid[idx])
            ax.imshow(image) 