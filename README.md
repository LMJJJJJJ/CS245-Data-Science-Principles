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

## How to Run these Code

### Project 1

*TBD*

### Project 2

*TBD*

### Project 3

*TBD*

### Project 4

*TBD*

## Some of the Experimental Results

### Project 1 Dimensionality Reduction (Feature Learning Part)

- Dataset: deep learning features of the AwA2 dataset [Download](http://cvml.ist.ac.at/AwA2/AwA2-features.zip)

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

- Dataset: deep learning features of the AwA2 dataset [Download](http://cvml.ist.ac.at/AwA2/AwA2-features.zip)

  - Note: These are features extracted using an ILSVRC-pretrained ResNet101

- For classification, 60% data are used for training, and the rest 40% are used for testing.

- Some of the results:

  | Method | Note                  | K in KNN | Test Acc. |
  | ------ | --------------------- | -------- | --------- |
  | LSML   | num. of samples 50Y^2 | 7        | 0.8951    |
  | NCA    | (None)                | 7        | 0.9092    |
  | LFDA   | reduce to 64 dim      | 7        | 0.9212    |


### Project 3 Feature Encoding

*TBD*

### Project 4 Domain Adaptation

*TBD*

