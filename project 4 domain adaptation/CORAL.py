"""
CORAL (AAAI 2016) Return of Frustratingly Easy Domain Adaptation

Adapted from: https://github.com/jindongwang/transferlearning/blob/master/code/traditional/CORAL/CORAL.py

Used for CS245 @ SJTU, 2020 Spring
"""

import numpy as np
import scipy.io
import scipy.linalg


class CORAL(object):
    def __init__(self):
        super(CORAL, self).__init__()

    def fit(self, Xs, Xt):
        '''
        Perform CORAL on the source domain features
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: New source domain features
        '''
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                         scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
        Xs_new = np.dot(Xs, A_coral)
        return Xs_new