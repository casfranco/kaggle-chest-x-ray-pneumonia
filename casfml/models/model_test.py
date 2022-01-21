import traceback
import tensorflow as tf
import os

#from classification_models.tfkeras import Classifiers
#import efficientnet.tfkeras as efn 


class ModelManager(object):
    """
    Class manager for diferent classification models 
    """
    #__classification_models__ = Classifiers.models_names() # models in classification_models library
    __effn_models__ = [('EfficientNetB'+ str(i)) for i in range(8)] #EfficientNetB0 --> EfficientNetB7
    __default_network__ = 'EfficientNetB0' # should be in the efn library


    def __init__():
        pass

    @staticmethod
    def load_model(hparams, path):
        model = ModelManager.create_model(hparams)
        model.load_weights( path )
        return model

    @staticmethod
    def create_model(hparams):
        '''method to call the appropiate model creation according to hparams configuration file''' 
        model_type = None

        # Available models and libraries
        switch={
            #'effn_model':ModelManager.__create_effn_model__, # from EfficientNet library
            #'cls_model':ModelManager.__create_cls_model__, # from classification models library
            'custom_model':ModelManager.__create_custom_model__, # Keras sequential model
            # Here can be inserted other functions to create keras models
            }

        # Check if suppported by Effn library
        if(hparams.model.lower() in map(str.lower,ModelManager.__effn_models__)):
            model_type = 'effn_model'
        
        # Check if suppported classification models library
        if(hparams.model.lower() in map(str.lower,ModelManager.__classification_models__)):
            model_type = 'cls_model'
        
        if(hparams.model == 'custom_model'):
            model_type = 'custom_model'

        model = None
        try:
            if( model_type in switch ):
                model = switch[model_type](hparams)
        except Exception as e:
            traceback.print_exc()
            model = None
        
        if( model is None):
            hparams.model = ModelManager.__default_network__            
            model = switch['effn_model'](hparams)

        return model

    # ---------------------------------------------------------------------------- #
    #                         functions for model creation                         #
    # ---------------------------------------------------------------------------- #


    @staticmethod
    def __create_custom_model__(hparams,n_channels=1): 
        ActivationRelu = tf.keras.activations.relu

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(16, (3, 3), activation=ActivationRelu, input_shape=(hparams.image_width, hparams.image_height, 1)))
        model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=ActivationRelu))
        model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=ActivationRelu))
        model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=ActivationRelu))
        model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation=ActivationRelu))
        model.add(tf.keras.layers.MaxPooling2D(pool_size = (2, 2)))

        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(activation = ActivationRelu, units = 512))
        model.add(tf.keras.layers.Dense(activation = ActivationRelu, units = 64))
        model.add(tf.keras.layers.Dense(activation = ActivationRelu, units = 16))
        model.add(tf.keras.layers.Dense(activation = "sigmoid", units = hparams.n_classes))
        
        return model
