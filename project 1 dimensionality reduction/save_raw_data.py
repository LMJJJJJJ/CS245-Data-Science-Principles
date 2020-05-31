import numpy as np
import time
from sklearn.model_selection import train_test_split

import os
import os.path as osp


def makedirs(dirs):
    if not osp.exists(dirs):
        os.makedirs(dirs)


def get_labels(filename="/home/data2/limingjie/data/Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt"):
    t = time.time()
    with open(filename, "r") as f:
        y = f.readlines()
    y = [int(label.strip()) for label in y]
    y = np.array(y)
    print("Labels loaded in {:.3f} seconds".format(time.time() - t))
    return y


def get_features(filename="/home/data2/limingjie/data/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt"):
    t = time.time()
    with open(filename, "r") as f:
        X = f.readlines()
    X = [[float(elem) for elem in line.split()] for line in X]
    X = np.array(X, dtype=np.float32)
    print("Features loaded in {:.3f} seconds".format(time.time() - t))
    return X


def get_data_info(X, y):
    assert X.shape[0] == y.shape[0]
    print("# Data Info:")
    print(" |- size: {}".format(X.shape[0]))
    print(" |- # dimension: {}".format(X.shape[1]))
    print(" |- # class: {}".format(y.max()-y.min()+1))


def get_data(feature_path="/home/data2/limingjie/data/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt",
             label_path="/home/data2/limingjie/data/Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt"):
    t = time.time()
    X = get_features(feature_path)
    y = get_labels(label_path)
    print("Data loaded in {:.3f} seconds".format(time.time() - t))
    return X, y

def save_data():
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    raw_data_folder = "./data/raw"
    makedirs(raw_data_folder)
    np.save(osp.join(raw_data_folder, "feature.npy"), X)
    np.save(osp.join(raw_data_folder, "label.npy"), y)

    split_data_folder = "./data/raw_split"
    makedirs(split_data_folder)
    np.save(osp.join(split_data_folder, "feature_train.npy"), X_train)
    np.save(osp.join(split_data_folder, "label_train.npy"), y_train)
    np.save(osp.join(split_data_folder, "feature_test.npy"), X_test)
    np.save(osp.join(split_data_folder, "label_test.npy"), y_test)


if __name__ == '__main__':
    save_data()