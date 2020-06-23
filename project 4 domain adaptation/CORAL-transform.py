from CORAL import CORAL

import numpy as np
import time
import argparse
import os
import os.path as osp
from sklearn.model_selection import train_test_split
from utils import makedirs

from sklearn.svm import SVC


def load_split(args):
    X_train = np.load(osp.join(args.data_root, 'feature_train.npy'))
    y_train = np.load(osp.join(args.data_root, 'label_train.npy'))
    X_test = np.load(osp.join(args.data_root, 'feature_test.npy'))
    y_test = np.load(osp.join(args.data_root, 'label_test.npy'))
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CORAL Transform")
    parser.add_argument('--data-root', default='./data/raw/Clipart_RealWorld')
    args = parser.parse_args()
    print(args)

    option = args.data_root.split("/")[-1].strip()
    save_root = osp.join("./data/CORAL", option)
    makedirs(save_root)

    X_train, X_test, y_train, y_test = load_split(args)
    print(X_train.shape)

    coral = CORAL()
    X_train_new = coral.fit(X_train, X_test)

    np.save(osp.join(save_root, "feature_train.npy"), X_train_new)
    np.save(osp.join(save_root, "label_train.npy"), y_train)
    np.save(osp.join(save_root, "feature_test.npy"), X_test)
    np.save(osp.join(save_root, "label_test.npy"), y_test)

