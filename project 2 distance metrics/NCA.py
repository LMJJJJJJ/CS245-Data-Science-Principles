import time
import argparse
import os
import os.path as osp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
from metric_learn import NCA


def makedirs(dirs):
    if not osp.exists(dirs):
        os.makedirs(dirs)


def load_split(args):
    X_train = np.load(osp.join(args.data_root, 'feature_train.npy'))
    y_train = np.load(osp.join(args.data_root, 'label_train.npy'))
    X_test = np.load(osp.join(args.data_root, 'feature_test.npy'))
    y_test = np.load(osp.join(args.data_root, 'label_test.npy'))
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser("NCA")
    parser.add_argument('--data-root', default='./data/raw_split')
    parser.add_argument('--n-components', type=int, default=2)
    parser.add_argument('--max-iter', type=int, default=100)
    args = parser.parse_args()

    name = f"{args.n_components}_{args.max_iter}"
    data_save_folder = f"./data/NCA/{name}"
    makedirs(data_save_folder)

    X_train, X_test, y_train, y_test = load_split(args)
    print(X_train.shape)

    t = time.time()

    nca = NCA(n_components=args.n_components, max_iter=args.max_iter, verbose=1)
    nca.fit(X_train, y_train)
    print(" # NCA fit done.")


    np.save(osp.join(data_save_folder, "feature_train.npy"), nca.transform(X_train))
    np.save(osp.join(data_save_folder, "label_train.npy"), y_train)
    np.save(osp.join(data_save_folder, "feature_test.npy"), nca.transform(X_test))
    np.save(osp.join(data_save_folder, "label_test.npy"), y_test)
