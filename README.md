# CS245-Data-Science-Principles
Some of the codes of projects in CS245@SJTU, 2020 Spring

## Requirements

All Experiments are conducted on a Linux Server with Four 4-core Intel(R) Xeon(R) CPU E5-2637 v4 @ 3.50GHz CPUs, 128G memory and 4 NVIDIA GeForce RTX 2080Ti GPUs.

- Python 3.6
- scikit-learn
- pytorch
- matplotlib
- numpy
- tqdm
- argparse

## Project 1 Dimensionality Reduction (Feature Learning Part)

- Dataset: deep learning features of the AwA2 dataset [Download](http://cvml.ist.ac.at/AwA2/AwA2-features.zip)

  - Note: These are features extracted using an ILSVRC-pretrained ResNet101

- For classification, 60% data are used for training, and the rest 40% are used for testing.

- Some of the results:

  | Method      | Dim. | SVM kernel | Train Acc. | Test Acc. |
  | ----------- | ---- | ---------- | ---------- | --------- |
  | AutoEncoder | 128  | linear     | 0.991      | 0.91      |
  | AutoEncoder | 256  | rbf        | 0.976      | 0.931     |
  | LLE (K=4)   | 128  | linear     | 0.117      | 0.112     |
  | LLE (K=128) |      |            |            |           |
  |             |      |            |            |           |
  |             |      |            |            |           |
  |             |      |            |            |           |
  |             |      |            |            |           |
  |             |      |            |            |           |

  