'''

Three methods of feature encoding (by Shangning Xu @xushangning)
 1. Bag of Words (BoW)
 2. Vector of Locally Aggregated Descriptors (VLAD)
 3. Fisher Vector

'''

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class Encoder(ABC):
    @abstractmethod
    def transform(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """
        :param X: Sequence of length n_samples, where each element is a Numpy
            array of shape (n_descriptors, n_features)
        :return: (n_samples, codebook_size)
        """
        pass


class BoW(Encoder):
    def __init__(self, knn: KMeans):
        self.knn = knn

    def transform(self, X: Sequence[np.ndarray]) -> np.ndarray:
        return np.array([
            np.bincount(self.knn.predict(x), minlength=self.knn.n_clusters) for x in X
        ])


class VLAD(Encoder):
    def __init__(self, knn: KMeans):
        self.knn = knn

    def transform(self, X: Sequence[np.ndarray]) -> np.ndarray:
        n_features_in_ = self.knn.cluster_centers_[0].shape[0]
        ret = []
        for x in X:
            labels = self.knn.predict(x)
            encoded = np.zeros(
                (self.knn.n_clusters, n_features_in_),
                dtype=X[0].dtype
            )
            for descriptor, label in zip(x, labels):
                encoded[label] += descriptor - self.knn.cluster_centers_[label]
            ret.append(encoded.flatten())
        return np.array(ret)


class Fisher(Encoder):
    def __init__(self, gmm):
        """
        :param n_components: number of components for GaussianMixture
        """
        self.gmm = gmm

    # def fit(self, X: np.ndarray):
    #     """
    #     :param X: shape (n_descriptors, n_features)
    #     :return: self
    #     """
    #     self.gmm.fit(X)
    #     return self

    def transform(self, X: Sequence[np.ndarray]) -> np.ndarray:
        ret = []
        for x in X:
            # x:                (n_descriptors, n_features)
            # means_:           (n_components, n_features)
            # covariances_:     (n_components,)
            # decentralized:    (n_components, n_descriptors, n_features)
            decentralized = (x[np.newaxis, ...] - self.gmm.means_[:, np.newaxis, :])\
                            / self.gmm.covariances_[:, np.newaxis, np.newaxis]
            # prob: (n_components, 1, n_descriptors)
            prob = self.gmm.predict_proba(x).T[:, np.newaxis, :]

            # sqrt_weights: (n_components, 1, 1)
            sqrt_weights = np.sqrt(self.gmm.weights_)[:, np.newaxis]
            # F_mu, F_sigma: (n_components, n_features)
            F_mu = np.squeeze(prob @ decentralized, axis=1)\
                   / x.shape[0] / sqrt_weights
            F_sigma = np.squeeze(prob @ (np.square(decentralized) - 1), axis=1)\
                      / x.shape[0] / np.sqrt(2) / sqrt_weights
            ret.append(np.stack((F_mu.flatten(), F_sigma.flatten())).flatten(order='F'))
        return np.array(ret)


if __name__ == '__main__':
    img = cv.imread('./data/Animals_with_Attributes2/JPEGImages/bat/bat_10001.jpg', cv.IMREAD_GRAYSCALE)
    sift = cv.xfeatures2d.SIFT_create(200)
    descriptors = sift.detectAndCompute(img, None)[1]
    # kmeans = KMeans().fit(descriptors)
    # bow = BoW(kmeans)
    # print(bow.transform((descriptors,)).shape)
    # vlad = VLAD(KMeans().fit(descriptors))
    # print(vlad.transform((descriptors[:100],)).shape)
    gmm = GaussianMixture(8, covariance_type='spherical')
    gmm.fit(descriptors)
    fisher = Fisher(gmm)
    print(fisher.transform((descriptors[:100],)).shape)