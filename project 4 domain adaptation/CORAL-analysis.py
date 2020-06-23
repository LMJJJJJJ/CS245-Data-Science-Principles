'''

Analyze the behavior of CORAL, including the similarity of the covariance matrix (also visualization)

'''

import numpy as np
import argparse
import os
import os.path as osp
from utils import makedirs
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image


def load_split(args):
    X_train = np.load(osp.join(args.data_root, 'feature_train.npy'))
    y_train = np.load(osp.join(args.data_root, 'label_train.npy'))
    X_test = np.load(osp.join(args.data_root, 'feature_test.npy'))
    y_test = np.load(osp.join(args.data_root, 'label_test.npy'))
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CORAL Analysis")
    parser.add_argument('--data-root', default='./data/CORAL/Art_RealWorld')
    args = parser.parse_args()
    print(args)

    datatype = args.data_root.split("/")[-2].strip()
    option = args.data_root.split("/")[-1].strip()
    save_root = osp.join("./analysis", datatype, option)
    makedirs(save_root)

    X_train, X_test, y_train, y_test = load_split(args)

    cov_source = np.cov(X_train.T)
    cov_target = np.cov(X_test.T)

    matplotlib.image.imsave(osp.join(save_root, "source.png"),
                            cov_source, cmap="seismic", vmin=-0.1, vmax=0.1)
    matplotlib.image.imsave(osp.join(save_root, "target.png"),
                            cov_target, cmap="seismic", vmin=-0.1, vmax=0.1)

    print("Square Error:", np.sum(np.square(cov_source - cov_target)))
    # print(cov_source.shape, np.min(cov_target), np.max(cov_target), np.mean(np.abs(cov_target)))
