"""
This module contains models for resent50
It's but an example. Modify it as you wish.
"""

import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D , BatchNormalization
from tensorflow.keras.models import Sequential, Model


class Resnet50:
    def __init__(self, image_size, channels = 3):

        self.input_shape = (image_size[0], image_size[1], channels)

    def get_model(self) -> Model:

        # get the pretrained model
        base_model = tf.keras.applications.ResNet50(input_shape=self.input_shape,
                                                    include_top=False,
                                                    weights='imagenet')
        base_model.trainable = False

        # batch normalization become trainable
        for layer in base_model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False
        base_model.summary()
        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(4, activation='softmax'))
        model.summary()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        return model
