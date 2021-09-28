import os
from os.path import join
import zipfile
import cv2
import numpy as np
import pandas as pd
from deep_utils import split_dir_of_dir
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import shutil


def preprocessing(kaggle_user="monibaravan",
                  kaggle_key="7a03bfd58536ced3a30c3b5c742096dd",
                  val_size=0.2,
                  augmented_samples=True,
                  augmented_simple_test=False,
                  remove_minor_class=True,
                  train_path='data/train',
                  val_path='data/val',
                  test_path='data/test'):
    print('If you are living in Iran, turn on you VPN!!!\n')
    os.environ['KAGGLE_USERNAME'] = kaggle_user  # username from the json file
    os.environ['KAGGLE_KEY'] = kaggle_key  # key from the json file
    dataset_path = 'blood-cells.zip'
    if not os.path.exists(dataset_path):
        os.system('kaggle datasets download -d paultimothymooney/blood-cells')
    else:
        print('Dataset exists')
    dataset_type, image_path = ('dataset2-master', "images") if augmented_samples else ('dataset-master', "JPEGImages")
    data_path = f"{dataset_type}/{dataset_type}/{image_path}"
    label_csv_path = f"{dataset_type}/{dataset_type}/labels.csv"

    with zipfile.ZipFile(dataset_path) as file:
        file.extractall()

    labels_df = pd.read_csv(label_csv_path)
    labels_df = labels_df.dropna(subset=['Image', 'Category'])
    labels_df['Image'] = labels_df['Image'].apply(
        lambda x: 'BloodImage_0000' + str(x) + '.jpg'
        if x < 10
        else ('BloodImage_00' + str(x) + '.jpg' if x > 99 else 'BloodImage_000' + str(x) + '.jpg')
    )
    labels_df = labels_df[['Image', 'Category']]

    labels_df = remove_minor_classes(labels_df, remove_minor_class)

    classes = labels_df.Category.unique()
    train_labels_df, val_test = train_test_split(labels_df, test_size=val_size * 2)
    val_labels_df, test_labels_df = train_test_split(val_test, test_size=0.5)
    if augmented_samples:
        os.system(f'mv -rf {test_path}; mkdir -p {test_path}')
        split_dir_of_dir(join(data_path, 'TRAIN'), train_dir=train_path, val_dir=val_path, test_size=val_size, remove_out_dir=True)
        main_test_path = join(data_path, 'TEST_SIMPLE' if augmented_simple_test else "TEST")
        os.system(f'mv {main_test_path}/* {test_path}')
    else:
        move_samples(data_path, train_labels_df, train_path, classes)
        move_samples(data_path, val_labels_df, val_path, classes)
        move_samples(data_path, test_labels_df, test_path, classes)

    shutil.rmtree('dataset-master')
    shutil.rmtree('dataset2-master')

    print('preprocessing is done!')


def remove_minor_classes(labels_df, remove_minor_class):
    # remove the minor class
    if remove_minor_class:
        cat_counts = labels_df.Category.value_counts()
        to_remove = cat_counts[cat_counts < 10].index
        labels_df = labels_df.replace(to_remove, np.nan).dropna()
    return labels_df


def move_samples(data_path, labels_df, path, classes):
    for cls in classes:
        tmp = join(path, cls)
        os.system(f"rm -rf {tmp}; mkdir -p {tmp}")
    for index, row in labels_df.iterrows():
        load_path = os.path.join(data_path, row['Image'])
        if not os.path.exists(load_path):
            continue

        img = cv2.imread(load_path)  # Load each Image
        cls = row['Category'].strip()  # Label
        save_path = os.path.join(path, cls, row['Image'])  # save path
        cv2.imwrite(save_path, img)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--kaggle_user", default="monibaravan", type=str)
    parser.add_argument("--kaggle_key", default="7a03bfd58536ced3a30c3b5c742096dd", type=str)
    parser.add_argument("--val_size", default=0.2, type=float)
    parser.add_argument("--augmented_samples", default=True, type=bool, help='Which dataset to use, augmented or not!')
    parser.add_argument("--augmented_simple_test", default=False, type=bool,
                        help='Which test_dataset to choose, simple or augmentd!')
    parser.add_argument("--remove_minor_class", default=True, type=bool,
                        help='Remove minor classes to make dataset balanced')
    parser.add_argument('--train-path', type=str, default='data/train',
                        help='Path to folder containing train dataset directory.',
                        required=False)
    parser.add_argument('--val-path', type=str, default='data/val',
                        help='Path to folder containing val dataset directory.',
                        required=False)
    parser.add_argument('--test-path', type=str, default='data/test',
                        help='Path to folder containing test dataset directory.',
                        required=False)
    args = parser.parse_args()
    preprocessing(
        kaggle_user=args.kaggle_user,
        kaggle_key=args.kaggle_key,
        val_size=args.val_size,
        augmented_samples=args.augmented_samples,
        remove_minor_class=args.remove_minor_class,
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        augmented_simple_test=args.augmented_simple_test
    )
