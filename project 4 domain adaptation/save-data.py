import os
import os.path as osp
import numpy as np
from utils import makedirs


def read_file(file, verbose=0):
    if verbose != 0:
        print()
        print("[read_file] Reading", file, "...")
    with open(file, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].split(",")
            lines[i] = [float(elem) for elem in lines[i]]
    lines = np.array(lines)
    features = lines[:, :-1]
    labels = lines[:, -1].astype(np.int64)
    if verbose != 0:
        print("[read_file]", "Finish reading", file)
        print("[read_file]", "features:", features.shape, features.dtype)
        print("[read_file]", "labels:", labels.shape, labels.dtype)
    return features, labels


def save_data(source, target, data_root="./data/Office-Home_resnet50", verbose=0):
    if verbose != 0:
        print()
        print("[save_data]", "Source:", source, "Target:", target)
    save_folder = osp.join("./data", f"{source}_{target}")
    makedirs(save_folder)
    train_file = osp.join(data_root, f"{source}_{source}.csv")
    test_file = osp.join(data_root, f"{source}_{target}.csv")
    X_train, y_train = read_file(train_file, verbose)
    X_test, y_test = read_file(test_file, verbose)
    if verbose != 0:
        print("[save_data]", "Saving data to", save_folder)
    np.save(osp.join(save_folder, "feature_train.npy"), X_train)
    np.save(osp.join(save_folder, "label_train.npy"), y_train)
    np.save(osp.join(save_folder, "feature_test.npy"), X_test)
    np.save(osp.join(save_folder, "label_test.npy"), y_test)
    if verbose != 0:
        print("[save_data] Finished!")


if __name__ == '__main__':
    source_list = ["Art", "Clipart", "Product"]
    target_list = ["RealWorld"] * 3
    for source, target in zip(source_list, target_list):
        save_data(source, target, verbose=1)