import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from models import load_model
from PIL import Image


class Detect:
    def __init__(self, model_name, weight_path, **kwargs):
        self.model = load_model(model_name=model_name, compile= True, **kwargs)
        self.model.load_weights(weight_path)
        self.input_shape = kwargs.get('input_shape')

    def detect(self, img):
        # apply necessary preprocessing

        img = image.load_img(img, target_size= self.input_shape)
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        result = self.model.predict(img)

        # apply necessary post-processing


        # return the results
        return result

    def detect_from_path(self, img_path):
        # make necessary modifications
        img = Image.open(img_path)
        return self.detect(img)
