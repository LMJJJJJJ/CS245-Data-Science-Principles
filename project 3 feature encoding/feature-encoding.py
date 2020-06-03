'''

Feature Encoding using BoW, VLAD, Fisher vector (2020.5.30)

 - Step 1: learn the codebook by sampling N*n_cluster local features in each class
 - Step 2: generate the global feature based on the codebook and local features
 - Step 3: generate the data for training the svm for classification

'''

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import argparse
import os
import os.path as osp

from utils import makedirs
from encode import BoW, VLAD, Fisher


def read_folder(folder):
    print("Reading", folder, "...")
    data = None
    for file in os.listdir(folder):
        file = osp.join(folder, file)
        if data is None:
            data = np.load(file)
        else:
            data = np.concatenate((data, np.load(file)), axis=0)
    return data


def encode_folder(folder, encoder, idx):
    print("Encoding", folder, '...')
    data = None
    for file in os.listdir(folder):
        file = osp.join(folder, file)
        encoded = encoder.transform((np.load(file),))
        if data is None:
            data = encoded
        else:
            data = np.concatenate((data, encoded), axis=0)
    label = np.array([idx] * data.shape[0])
    return data, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Feature Encoding")
    parser.add_argument('--method', type=str, default='fisher')
    parser.add_argument('--n-cluster', type=int, default=128)
    parser.add_argument('--local-dir', type=str, default='./save-sift/10')
    parser.add_argument('--N', type=int, default=5)
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()
    print(args)

    train_folder = osp.join(args.local_dir, 'descriptor/train')
    test_folder = osp.join(args.local_dir, 'descriptor/test')
    animals = os.listdir(train_folder)
    samples_per_class = args.N * args.n_cluster
    save_folder = osp.join('save-encoded', "-".join(args.local_dir.split('/')[-2:]),
                           f"{args.method}-{args.n_cluster}-{args.N}")
    makedirs(save_folder)

    # 1. sample from the training set
    codebook_data = None
    for i in range(len(animals)):
        data = read_folder(osp.join(train_folder, animals[i]))
        np.random.shuffle(data)
        data = data[:samples_per_class]
        if codebook_data is None:
            codebook_data = data
        else:
            codebook_data = np.concatenate((codebook_data, data), axis=0)

    # 2. learn the codebook and encode the local features into global features
    encoder = None
    if args.method == 'bow':
        print("Learning the codebook ...")
        kmeans = KMeans(n_clusters=args.n_cluster, verbose=args.verbose)
        kmeans.fit(codebook_data)
        encoder = BoW(kmeans)
    elif args.method == 'vlad':
        print("Learning the codebook ...")
        kmeans = KMeans(n_clusters=args.n_cluster, verbose=args.verbose)
        kmeans.fit(codebook_data)
        encoder = VLAD(kmeans)
    elif args.method == 'fisher':
        print("Learning the codebook ...")
        gmm = GaussianMixture(args.n_cluster, covariance_type='spherical')
        gmm.fit(codebook_data)
        encoder = Fisher(gmm)
    else:
        print(f"[Fail] Method {args.method} not supported yet.")

    if encoder is None:
        print(f"[Fail] WTF?")
    # 3. Encode the training set
    features = None
    labels = None
    for i in range(len(animals)):
        folder = osp.join(train_folder, animals[i])
        feature, label = encode_folder(folder, encoder, i)
        if features is None:
            features = feature
            labels = label
        else:
            features = np.concatenate((features, feature), axis=0)
            labels = np.concatenate((labels, label), axis=0)
    np.save(osp.join(save_folder, 'feature_train.npy'), features)
    np.save(osp.join(save_folder, 'label_train.npy'), labels)
    del features
    del labels
    # 4. Encode the test set
    features = None
    labels = None
    for i in range(len(animals)):
        folder = osp.join(test_folder, animals[i])
        feature, label = encode_folder(folder, encoder, i)
        if features is None:
            features = feature
            labels = label
        else:
            features = np.concatenate((features, feature), axis=0)
            labels = np.concatenate((labels, label), axis=0)
    np.save(osp.join(save_folder, 'feature_test.npy'), features)
    np.save(osp.join(save_folder, 'label_test.npy'), labels)




