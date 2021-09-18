import kaggle
import os
import zipfile
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import shutil


def preprocessing(kaggle_user="monibaravan",
                  kaggle_key="7a03bfd58536ced3a30c3b5c742096dd",
                  dataset_path='blood-cells.zip',
                  data_path="dataset-master/dataset-master/JPEGImages",
                  label_csv_path="dataset-master/dataset-master/labels.csv",
                  des_path="", # defualt = empty(Local file)
                  test_size=0.3):
    print('If you are living in Iran, turn on you VPN!!!\n')
    os.environ['KAGGLE_USERNAME'] = kaggle_user  # username from the json file
    os.environ['KAGGLE_KEY'] = kaggle_key  # key from the json file
    os.system('kaggle datasets download -d paultimothymooney/blood-cells')

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
    cat_counts = labels_df.Category.value_counts()
    to_remove = cat_counts[cat_counts < 10].index
    labels_df = labels_df.replace(to_remove, np.nan).dropna()
    classes = labels_df.Category.unique()
    idx_classes = {}
    for i, cls in enumerate(classes):
        idx_classes[cls] = i
    # Split train and test
    train_df, test_df = train_test_split(
        labels_df,
        test_size=test_size,
        stratify=labels_df['Category']
    )

    # Saving Data
    os.makedirs(os.path.join(des_path, 'dataset'))
    os.makedirs(os.path.join(des_path, 'dataset', 'train'))
    # os.makedirs(os.path.join(des_path,'dataset', 'validation'))
    os.makedirs(os.path.join(des_path, 'dataset', 'test'))

    for cls in classes:
        os.makedirs(os.path.join(des_path, 'dataset', 'train', cls))
        # os.makedirs(os.path.join(des_path,'dataset', 'validation', cls))
        os.makedirs(os.path.join(des_path, 'dataset', 'test', cls))

    # TRAIN DATA
    for index, row in train_df.iterrows():
        load_path = os.path.join(data_path, row['Image'])
        if not os.path.exists(load_path):
            continue

        img = cv2.imread(load_path)  # Load each Image
        cls = row['Category'].strip()  # Label
        save_path = os.path.join(des_path, 'dataset', 'train', cls, row['Image'])  # save path
        # print(row['Image'], cls)
        cv2.imwrite(save_path, img)  # save each image to its Category

    # Test DATA
    for index, row in test_df.iterrows():
        load_path = os.path.join(data_path, row['Image'])
        if not os.path.exists(load_path):
            continue

        img = cv2.imread(load_path)  # Load each Image
        cls = row['Category'].strip()  # Label
        save_path = os.path.join(des_path, 'dataset', 'test', cls, row['Image'])  # save path
        # print(row['Image'], cls)
        cv2.imwrite(save_path, img)  # save each image to its Category

    # Removing extra files
    shutil.rmtree('dataset-master')
    shutil.rmtree('dataset2-master')
    os.remove(dataset_path)


    print('preprocessing is done!')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--kaggle_user", default="monibaravan", type=str)
    parser.add_argument("--kaggle_key", default="7a03bfd58536ced3a30c3b5c742096dd", type=str)
    parser.add_argument("--dataset_path", default='blood-cells.zip', type=str)
    parser.add_argument("--data_path", default="dataset-master/dataset-master/JPEGImages", type=str)
    parser.add_argument("--label_csv_path", default="dataset-master/dataset-master/labels.csv", type=str)
    parser.add_argument("--des_path", default="", type=str)
    parser.add_argument("--test_size", default=0.3, type=float)
    args = parser.parse_args()
    preprocessing(
        kaggle_user=args.kaggle_user,
        kaggle_key=args.kaggle_key,
        dataset_path=args.dataset_path,
        data_path=args.data_path,
        label_csv_path=args.label_csv_path,
        des_path=args.des_path,
        test_size=args.test_size,
    )
