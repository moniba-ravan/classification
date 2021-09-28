import math
import os
import numpy as np
import tensorflow as tf
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,
    ToFloat, ShiftScaleRotate, Sharpen, Emboss, HorizontalFlip, PiecewiseAffine,
    ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
import cv2


"""
CREATING CUSTOM DATA GENERATOR
"""


class CustomImageGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 aug_prop=0.75,
                 img_dir='data/data/train',
                 batch_size=32,
                 img_size=(224, 224),
                 n_channels=3,
                 augment=False,
                 shuffle=True):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.aug_prob = aug_prop
        self.aug_func = self.aug_func(self.aug_prob)
        self.augment = augment
        self.img_ids, self.labels, self.class_name_map = self.get_label_list(img_dir)
        self.indexes = None
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.floor(len(self.img_ids) / self.batch_size)

    def get_image(self, name):
        img = cv2.imread(name)[..., ::-1]
        img = cv2.resize(img, self.img_size)
        return img

    def __data_generation(self, list_id_temp, labels):
        x = np.zeros((self.batch_size, *self.img_size, self.n_channels))
        y = np.zeros(self.batch_size, dtype=float)
        for i, (name, label) in enumerate(zip(list_id_temp, labels)):
            img = self.get_image(name)
            if self.augment:
                img = self.aug_func(self.aug_prob)(image=img)
            x[i] = img
            y[i] = self.class_name_map[label]

        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        list_id_temp = [self.img_ids[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]
        x, y = self.__data_generation(list_id_temp, labels)
        return x, y

    @staticmethod
    def get_label_list(img_dir):
        class_name_map = {class_name: en for en, class_name in enumerate(os.listdir(img_dir))}
        labels = []
        img_list = []
        for class_name in os.listdir(img_dir):
            class_dir = os.path.join(img_dir, class_name)
            label = class_name
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img_list.append(img_path)
                labels.append(label)
        return img_list, labels, class_name_map

    @staticmethod
    def aug_func(p=0.75):
        return Compose([Flip(p=0.5),
                        GaussNoise(p=0.2),
                        OneOf([
                            MotionBlur(p=.2),
                            MedianBlur(blur_limit=3, p=.1),
                            Blur(blur_limit=3, p=.1),
                        ], p=0.2),
                        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
                        OneOf([
                            OpticalDistortion(p=0.3),
                            GridDistortion(p=.1),
                            PiecewiseAffine(p=0.3),
                        ], p=0.2),
                        OneOf([
                            Sharpen(),
                            Emboss(),
                            RandomContrast(),
                            RandomBrightness(),
                        ], p=0.3),
                        HueSaturationValue(p=0.3),
                        ], p=p)


def get_loader(train_path='data/data/train',
               val_path='data/data/val',
               test_path='data/data/test',
               batch_size=32,
               target_size=(224, 224),
               aug_prop=0.75
               ):
    train_gen = CustomImageGenerator(aug_prop=aug_prop, img_dir=train_path, batch_size=batch_size, img_size=target_size)
    val_gen = CustomImageGenerator(aug_prop=aug_prop, img_dir=val_path, img_size=target_size)
    test_gen = CustomImageGenerator(aug_prop=0, img_dir=test_path, img_size=target_size)

    return train_gen, val_gen, test_gen


if __name__ == '__main__':
    train_gen, val_gen, test_gen = get_loader(train_path='data/train', val_path='data/val', test_path='data/test')
    for x, y in train_gen:
        print(x.shape, y.shape)
        break
