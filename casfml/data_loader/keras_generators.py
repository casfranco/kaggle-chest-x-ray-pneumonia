from tensorflow.keras.preprocessing.image import ImageDataGenerator


class KerasClsGenerators(object):
    def __init__():
        pass

    @staticmethod
    def get_generator(batch_size=1, image_size=(256, 256), df=None, data_augmentation=None, shuffle=False):

        if(data_augmentation is None):
            gen = ImageDataGenerator(rescale=1./255)

        else:
            gen = ImageDataGenerator(rescale=1./255, **data_augmentation)

        generator = gen.flow_from_dataframe(
            x_col='filepath',
            y_col='label',
            dataframe=df,
            # directory="train",
            class_mode='binary',
            # classes=None,
            color_mode='grayscale',
            batch_size=batch_size,
            target_size=image_size,
            shuffle=shuffle,
            seed=21)

        return generator
