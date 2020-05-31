import numpy as np
import time

from sklearn.svm import SVC


def load_split():
    X_train = np.load("./data/raw_split/feature_train.npy")
    y_train = np.load("./data/raw_split/label_train.npy")
    X_test = np.load("./data/raw_split/feature_test.npy")
    y_test = np.load("./data/raw_split/label_test.npy")
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_split()

    t = time.time()
    svm = SVC(C=1.0, verbose=1)
    svm.fit(X_train, y_train)
    train_acc = svm.score(X_train, y_train)
    test_acc = svm.score(X_test, y_test)

    print("Finished in {:.3f} seconds".format(time.time() - t))
    print("# Result (once):")
    print(" |- train accuracy: {:.3f}".format(train_acc))
    print(" |- test accuracy: {:.3f}".format(test_acc))