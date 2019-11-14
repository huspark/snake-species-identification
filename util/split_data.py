import argparse
import os
import random
from shutil import copy2
import sys

import pandas as pd



def process_arg():
    """
    Process the user input for data path.

    :return args: argparse.Namespace()
    """
    parser = argparse.ArgumentParser(description="This file organizes files in train_images folder.")
    parser.add_argument('-p', '--data_path', type=str, required=True,  help='the path containing train_images folder and train_labels.csv')
    parser.add_argument('-s', '--split_ratio', default=0.8, type=float, required=False,  help='train-valid split ratio')

    args = parser.parse_args()

    return args


def make_dirs(data_path, df):
    """
    Make train and valid folders in data path and create subdirectories with snake species.

    :param data_path: path containing the train_images folder and the train_labels.csv file
    :param df: dataframe from the train_labels.csv file
    :return: None
    """
    print('creating train/valid directories with subdirectories')
    if not os.path.exists(data_path + '/train'):
        os.mkdir(data_path + '/train')
    if not os.path.exists(data_path + '/valid'):
        os.mkdir(data_path + '/valid')

    for label in df['scientific_name'].unique():
        if not os.path.exists(data_path + '/train/' + label):
            os.mkdir(data_path + '/train/' + label)
        if not os.path.exists(data_path + '/valid/' + label):
            os.mkdir(data_path + '/valid/' + label)


def organize_imgs(data_path, df):
    """
    Organize images from train_images folder into the created train/valid folders.

    :param data_path: path containing the train_images folder and the train_labels.csv file
    :param df: dataframe from the train_labels.csv file
    :return:
    """
    print('copying images from train_images to train folders')
    num_imgs = len(os.listdir(train_img_path))

    for idx, f in enumerate(os.listdir(train_img_path)):
        if idx % 100 == 0 or idx == num_imgs - 1:
            print('copying files ' + str(idx) + '/' + str(num_imgs) + '...')

        if os.stat(train_img_path + '/' + f).st_size > 0:
            species = df[df['filename'] == f]['scientific_name'].values[0]
            copy2(train_img_path + '/' + f, data_path + '/train/' + species + '/' + f)


def split_data(data_path, split_ratio):
    """
    Split data into train and valid set.

    :param data_path: path containing the train_images folder and the train_labels.csv file
    :param split_ratio: train-valid split ratio
    :return:
    """
    print('splitting images into train/valid images')
    species_list = os.listdir(data_path + '/train')
    if '.DS_Store' in species_list:
        species_list.remove('.DS_Store')

    for species in species_list:
        train_dir = os.path.join(data_path + '/train', species)
        valid_dir = os.path.join(data_path + '/valid', species)

        imgs = os.listdir(train_dir)
        valid_imgs = random.sample(imgs, k=int(len(imgs)*(1-split_ratio)))
        for img in valid_imgs:
            src = train_dir + '/' + img
            dst = valid_dir + '/' + img
            if os.path.exists(src):
                os.rename(src, dst)


if __name__=='__main__':
    # Process arguments
    args = process_arg()

    # Check if the path is valid
    try:
        data_path = args.data_path
        train_img_path = data_path + '/train_images'
        df_train_labels = pd.read_csv(data_path + '/train_labels.csv')
    except FileNotFoundError:
        sys.exit("Error: Invalid path")

    split_ratio = args.split_ratio

    df_train_labels.drop(['hashed_id', 'country', 'continent'], axis=1, inplace=True)

    make_dirs(data_path=data_path, df=df_train_labels)
    organize_imgs(data_path=data_path, df=df_train_labels)
    split_data(data_path=data_path, split_ratio=split_ratio)

