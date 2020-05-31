import numpy as np
import time
import argparse
import os
import os.path as osp
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
import matplotlib.pyplot as plt


def load_raw(args):
    X = np.load(osp.join(args.data_root, 'feature.npy'))
    y = np.load(osp.join(args.data_root, 'label.npy'))
    return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test")
    parser.add_argument('--method', default='LLE')
    parser.add_argument('--method-root', default=None)
    parser.add_argument('--name', default="components_2_neighbors_4")
    parser.add_argument('--data-root', default=None)
    args = parser.parse_args()
    args.method_root = osp.join("./data", args.method)
    args.data_root = osp.join(args.method_root, args.name)

    X, y = load_raw(args)
    print(X.shape)

    x_min, x_max = X.min(0), X.max(0)
    X_normalized = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))
    for i in range(X_normalized.shape[0]):
        plt.text(X_normalized[i, 0], X_normalized[i, 1], str(y[i]), color=plt.cm.Set3(y[i] % 12),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(osp.join("fig", args.method, f"{args.name}.pdf"))