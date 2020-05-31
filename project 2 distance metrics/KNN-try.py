import numpy as np
import time
import argparse
import os
import os.path as osp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

def load_split(args):
    X_train = np.load(osp.join(args.data_root, 'feature_train.npy'))
    y_train = np.load(osp.join(args.data_root, 'label_train.npy'))
    X_test = np.load(osp.join(args.data_root, 'feature_test.npy'))
    y_test = np.load(osp.join(args.data_root, 'label_test.npy'))
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test")
    parser.add_argument('--data-root', default='./data/raw_split')
    parser.add_argument('--n-neighbors', type=int, default=5)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_split(args)
    print(X_train.shape)

    t = time.time()

    knn = KNeighborsClassifier(n_neighbors=args.n_neighbors, metric='minkowski', p=2, n_jobs=4)

    knn.fit(X_train, y_train)
    print("# KNN fit done.")
    # train_acc = accuracy_score(y_train, knn.predict(X_train))
    test_acc = accuracy_score(y_test, knn.predict(X_test))
    print("Finished in {:.3f} seconds".format(time.time() - t))
    print("# Result (once):")
    # print(" |- train accuracy: {:.3f}".format(train_acc))
    print(" |- test accuracy: {:.3f}".format(test_acc))



