import numpy as np
import time
import argparse
import os
import os.path as osp
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import preprocessing


def load_split(args):
    X_train = np.load(osp.join(args.data_root, 'feature_train.npy'))
    y_train = np.load(osp.join(args.data_root, 'label_train.npy'))
    X_test = np.load(osp.join(args.data_root, 'feature_test.npy'))
    y_test = np.load(osp.join(args.data_root, 'label_test.npy'))
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test")
    parser.add_argument('--data-root', default='./save-encoded/save-sift-10/vlad-128-5')
    parser.add_argument('--svm-c', type=float, default=1.0)
    parser.add_argument('--kernel', type=str, default="rbf")
    parser.add_argument('--reduce', type=int, default=-1)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_split(args)
    print(X_train.shape)

    print("Normalizing data (L2 Normalization) ...")
    X_train = preprocessing.normalize(X_train, norm='l2')
    X_test = preprocessing.normalize(X_test, norm='l2')

    if args.reduce > 0:
        print(f"Reduce to {args.reduce}-dim by PCA ...")
        pca = PCA(n_components=args.reduce)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

    t = time.time()
    svm = SVC(C=args.svm_c, kernel=args.kernel, verbose=0)
    svm.fit(X_train, y_train)
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_test, y_test)

    print("Finished in {:.3f} seconds".format(time.time() - t))
    print("# Result (once):")
    print(" |- train accuracy: {:.3f}".format(train_acc))
    print(" |- test accuracy: {:.3f}".format(test_acc))