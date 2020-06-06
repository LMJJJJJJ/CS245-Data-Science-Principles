import numpy as np
import time
import argparse
import os
import os.path as osp
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC


def load_split(args):
    X_train = np.load(osp.join(args.data_root, 'feature_train.npy'))
    y_train = np.load(osp.join(args.data_root, 'label_train.npy'))
    X_test = np.load(osp.join(args.data_root, 'feature_test.npy'))
    y_test = np.load(osp.join(args.data_root, 'label_test.npy'))
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test")
    parser.add_argument('--data-root', default='./data/Product_RealWorld')
    parser.add_argument('--svm-c', type=float, default=1.0)
    parser.add_argument('--kernel', type=str, default="rbf")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_split(args)
    print(X_train.shape)

    t = time.time()
    svm = SVC(C=args.svm_c, kernel=args.kernel)
    svm.fit(X_train, y_train)
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_test, y_test)

    print("Finished in {:.3f} seconds".format(time.time() - t))
    print("# Result (once):")
    print(" |- train accuracy: {:.3f}".format(train_acc))
    print(" |- test accuracy: {:.3f}".format(test_acc))