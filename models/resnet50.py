"""
This module contains models for resent50
It's but an example. Modify it as you wish.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.resnet50 import preprocess_input


class Resnet50:
    def __init__(self, image_size=(200, 200), n_classes=4, fine_tune=False):
        self.input_shape = (image_size[0], image_size[1], 3)
        self.n_classes = n_classes
        self.fine_tune = fine_tune

    def get_model(self) -> Model:

        input = Input(self.input_shape)
        x = preprocess_input(input)

        # get the pretrained model
        base_model = tf.keras.applications.ResNet50(input_shape=self.input_shape,
                                                    include_top=False,
                                                    weights='imagenet')
        for layer in base_model.layers:
            layer.trainable = False

        # base_model.summary()
        x = base_model(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128)(x)
        x = Dropout(0.2)(x)
        x = Dense(self.n_classes, activation='softmax')(x)
        model = Model(input, x)
        if self.fine_tune:
            for layer in model.layers:
                layer.trainable = True

        model.summary()
        return model
