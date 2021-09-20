from typing import Tuple
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input


def get_loader(dataset_path,
               valid_size,
               batch_size,
               target_size
               ) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
    dataset_dir = os.path.join(dataset_path, 'dataset')

    # train & validation dataset
    data_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        # rotation_range=30,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # brightness_range=(0.8, 1.2),
        # shear_range=50,
        # zoom_range=0.2,
        # fill_mode = 'constant',
        # cval=255,
        # horizontal_flip = True,
        # vertical_flip = True,
        # rescale=1/255.,
        validation_split=valid_size,
    )

    train_gen = data_gen.flow_from_directory(os.path.join(dataset_dir, 'train'),
                                             target_size=target_size,
                                             batch_size=batch_size,
                                             class_mode='categorical',
                                             subset='training')

    valid_gen = data_gen.flow_from_directory(os.path.join(dataset_dir, 'train'),
                                             target_size=target_size,
                                             batch_size=batch_size,
                                             class_mode='categorical',
                                             subset='validation')

    # test dataset
    data_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        # rotation_range=30,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # brightness_range=(0.8, 1.2),
        # shear_range=50,
        # zoom_range=0.6,
        # fill_mode = 'constant',
        # cval=255,
        # horizontal_flip = True,
        # vertical_flip = True,
        # rescale=1/255.,
    )

    test_gen = data_gen.flow_from_directory(os.path.join(dataset_dir, 'test'),
                                            target_size=target_size,
                                            batch_size=batch_size,
                                            class_mode='categorical'
                                            )

    return train_gen, valid_gen, test_gen
