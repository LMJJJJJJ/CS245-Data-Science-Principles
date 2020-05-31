import numpy as np
import time
from sklearn.model_selection import train_test_split

import os
import os.path as osp
import argparse


def makedirs(dirs):
    if not osp.exists(dirs):
        os.makedirs(dirs)


def load_split(args):
    X = np.load(osp.join(args.data_root, 'feature.npy'))
    y = np.load(osp.join(args.data_root, 'label.npy'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test")
    parser.add_argument('--data-root', default='./data/raw')
    args = parser.parse_args()

    split_data_folder = "./data/raw_split"
    makedirs(split_data_folder)

    X_train, X_test, y_train, y_test = load_split(args)
    print(X_train.shape)
    print(X_test.shape)
    np.save(osp.join(split_data_folder, "feature_train.npy"), X_train)
    np.save(osp.join(split_data_folder, "label_train.npy"), y_train)
    np.save(osp.join(split_data_folder, "feature_test.npy"), X_test)
    np.save(osp.join(split_data_folder, "label_test.npy"), y_test)