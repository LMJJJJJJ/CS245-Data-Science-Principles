import os
import os.path as osp
import argparse
import pickle
import numpy as np


def makedirs(dirs):
    if not osp.exists(dirs):
        os.makedirs(dirs)


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_data(source, target, verbose=0):
    if verbose != 0:
        print()
        print("[get_data]", "Source:", source, "Target:", target)
    folder = osp.join("data", f"{source}_{target}")
    if verbose != 0:
        print("[get_data]", "Reading data from", folder)
    X_train = np.load(osp.join(folder, "feature_train.npy"))
    X_test = np.load(osp.join(folder, "feature_test.npy"))
    y_train = np.load(osp.join(folder, "label_train.npy"))
    y_test = np.load(osp.join(folder, "label_test.npy"))
    if verbose != 0:
        print("Train:", X_train.shape, y_train.shape)
        print("Test:", X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data("Art", "RealWorld", verbose=1)


