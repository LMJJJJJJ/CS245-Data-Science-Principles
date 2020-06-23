from BDA import BDA

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
    parser = argparse.ArgumentParser("BDA Transform")
    parser.add_argument('--data-root', default='./data/raw/Art_RealWorld')
    parser.add_argument('--kernel-type', default='primal', help='primal | linear | rbf')
    parser.add_argument('--bda-dim', type=int, default=32)
    parser.add_argument('--lamb', type=float, default=1.0)
    parser.add_argument('--mu', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--mode', default='BDA', help='BDA | WBDA')
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--estimate', type=int, default=0)
    args = parser.parse_args()
    print(args)

    option = args.data_root.split("/")[-1].strip()
    save_root = osp.join("./data/BDA_{}_{}_{}_{}_{}_{}_{}_{}".format(args.kernel_type, args.bda_dim, args.lamb, args.mu,
                                                                     args.gamma, args.T, args.mode, args.estimate), option)
    makedirs(save_root)

    X_train, X_test, y_train, y_test = load_split(args)
    print(X_train.shape)

    bda = BDA(kernel_type=args.kernel_type, dim=args.bda_dim, lamb=args.lamb,
              mu=args.mu, gamma=args.gamma, T=args.T, mode=args.mode, estimate_mu=args.estimate)
    _, _, _, X_train_new, X_test_new = bda.fit_predict(X_train, y_train, X_test, y_test)

    np.save(osp.join(save_root, "feature_train.npy"), X_train_new)
    np.save(osp.join(save_root, "label_train.npy"), y_train)
    np.save(osp.join(save_root, "feature_test.npy"), X_test_new)
    np.save(osp.join(save_root, "label_test.npy"), y_test)