# CS245-Data-Science-Principles
Some of my codes of projects in CS245@SJTU, 2020 Spring

## Requirements

All Experiments are conducted on a Linux Server with Four 4-core Intel(R) Xeon(R) CPU E5-2637 v4 @ 3.50GHz CPUs, 128G memory and 4 NVIDIA GeForce RTX 2080Ti GPUs.

- Python 3.6
- scikit-learn
- pytorch
- matplotlib
- numpy
- tqdm
- argparse
- [metric-learn](http://contrib.scikit-learn.org/metric-learn/) : for project 2
- opencv: for project 3

## Some of the Experimental Results

### Project 1 Dimensionality Reduction (Feature Learning Part)

- Dataset: deep learning features of the AwA2 dataset (301 MB) [Download](http://cvml.ist.ac.at/AwA2/AwA2-features.zip)

  - Note: These are features extracted using an ILSVRC-pretrained ResNet101

- For classification, 60% data are used for training, and the rest 40% are used for testing.

- Some of the results:

  | Method           | Dim. | SVM kernel | Train Acc. | Test Acc. |
  | ---------------- | ---- | ---------- | ---------- | --------- |
  | AutoEncoder      | 128  | linear     | 0.991      | 0.91      |
  | AutoEncoder      | 256  | rbf        | 0.976      | 0.931     |
  | LLE (K=4)        | 128  | linear     | 0.117      | 0.112     |
  | LLE (K=64)       | 128  | rbf        | 0.933      | 0.908     |
  | t-SNE (exact)    | 2    |            |            | 0.875     |
  | t-SNE (BH)       | 2    |            |            | 0.869     |
  | t-SNE (exact)    | 3    |            |            | 0.867     |
  | t-SNE (BH)       | 3    |            |            | 0.878     |
  | LDA + t-SNE (BH) | 3    |            |            | 0.955     |
  | PCA + t-SNE (BH) | 3    |            |            | 0.875     |
  | t-SNE (exact)    | 16   |            |            | 0.778     |
  | MDS              | 2    | rbf        | 0.187      | 0.188     |
  | MDS              | 16   | rbf        | 0.753      | 0.734     |


### Project 2 KNN with Different Distance Metrics (LSML, NCA, LFDA)

- Dataset: deep learning features of the AwA2 dataset (301 MB) [Download](http://cvml.ist.ac.at/AwA2/AwA2-features.zip)

  - Note: These are features extracted using an ILSVRC-pretrained ResNet101

- For classification, 60% data are used for training, and the rest 40% are used for testing.

- Some of the results:

  | Method | Note                  | K in KNN | Test Acc. |
  | ------ | --------------------- | -------- | --------- |
  | LSML   | num. of samples 50Y^2 | 7        | 0.8951    |
  | NCA    | (None)                | 7        | 0.9092    |
  | LFDA   | reduce to 64 dim      | 7        | 0.9212    |


### Project 3 Feature Encoding

- Dataset: AwA2 dataset JPEG images (13 GB) [Download](https://cvml.ist.ac.at/AwA2/AwA2-data.zip)

- For classification, 60% data are used for training, and the rest 40% are used for testing.

- Some of the results by SIFT (test accuracy)

  | N_cluster | N_kp | Note                   | BoW   | VLAD  | Fisher |
  | --------- | ---- | ---------------------- | ----- | ----- | ------ |
  | 128       | 10   | VLAD Fisher PCA to 128 | 0.124 | 0.129 | 0.125  |
  | 128       | 100  | VLAD Fisher PCA to 128 | 0.208 | 0.257 | 0.219  |
  | 256       | 100  | VLAD Fisher PCA to 256 | 0.217 | 0.259 | 0.227  |
  | 512       | 100  | VLAD Fisher PCA to 512 | 0.223 | 0.242 | 0.235  |
  | 128       | 500  | VLAD Fisher PCA to 128 | 0.277 | 0.351 | 0.282  |
  | 256       | 500  | VLAD Fisher PCA to 256 | 0.288 | 0.353 | 0.298  |
  | 512       | 500  | VLAD Fisher PCA to 512 | 0.301 | 0.343 | 0.304  |

- Some of the results by Selective Search then CNN (test accuracy)

  | N_cluster | Pretrained CNN | Method   | Note                   | BoW   | VLAD  | Fisher |
  | --------- | -------------- | -------- | ---------------------- | ----- | ----- | ------ |
  | 128       | AwA2           | original | VLAD Fisher PCA to 128 | 0.695 | 0.687 | 0.089  |
  | 128       | AwA2           | topleft  | VLAD Fisher PCA to 128 | 0.699 | 0.685 | 0.122  |
  | 128       | AwA2           | resize   | VLAD Fisher PCA to 128 | 0.143 | 0.160 | 0.128  |
  | 128       | ImageNet       | original | VLAD Fisher PCA to 128 | 0.803 | 0.829 | 0.651  |
  | 128       | ImageNet       | topleft  | VLAD Fisher PCA to 128 | 0.822 | 0.833 | 0.329  |
  | 128       | ImageNet       | resize   | VLAD Fisher PCA to 128 | 0.173 | 0.242 | 0.153  |

  


### Project 4 Domain Adaptation

*TBD*

