import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import argparse

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding
from sklearn.decomposition import SparseCoder, DictionaryLearning

from mpl_toolkits.mplot3d import Axes3D

import os
import os.path as osp

def makedirs(dirs):
    if not osp.exists(dirs):
        os.makedirs(dirs)


def load_raw():
    X = np.load("./data/raw/feature.npy")
    y = np.load("./data/raw/label.npy")
    return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser("t-SNE decomposition")
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--init", type=str, default="pca")
    parser.add_argument("--method", type=str, default="barnes_hut")
    args = parser.parse_args()
    X, y = load_raw()

    name = f"{args.n_components}_{args.init}_{args.method}"
    data_save_folder = f"./data/tSNE/{name}"
    fig_save_folder = f"./fig/tSNE"
    makedirs(data_save_folder)
    makedirs(fig_save_folder)

    tSNE = TSNE(n_components=args.n_components, verbose=1, init=args.init, method=args.method)
    X_decomposed = tSNE.fit_transform(X)
    np.save(osp.join(data_save_folder, "feature.npy"), X_decomposed)
    np.save(osp.join(data_save_folder, "label.npy"), y)

    if args.n_components == 2:
        x_min, x_max = X_decomposed.min(0), X_decomposed.max(0)
        X_normalized = (X_decomposed - x_min) / (x_max - x_min)
        plt.figure(figsize=(8, 8))
        for i in range(X_normalized.shape[0]):
            plt.text(X_normalized[i, 0], X_normalized[i, 1], str(y[i]), color=plt.cm.Set3(y[i] % 12),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(osp.join(fig_save_folder, f"{name}.pdf"))