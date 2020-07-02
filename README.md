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

### Project 1 Dimensionality Reduction (Feature Learning Part) ([our report](http://bcmi.sjtu.edu.cn/home/niuli/teaching/2020_1_2.pdf))

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


### Project 2 KNN with Different Distance Metrics (LSML, NCA, LFDA) ([our report](http://bcmi.sjtu.edu.cn/home/niuli/teaching/2020_2_2.pdf))

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

- Dataset: Office-Home dataset (4 domains -- Art, Clipart, Product, Real-World)

  - Raw images: [Download](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view) (在Google Drive上，需要科学上网)
  - ResNet-50 pretrained features: [Download](https://pan.baidu.com/s/1qvcWJCXVG8JkZnoM4BVoGg) (在百度云上)

- Only consider 3 cases -- Art to Real-World, Clipart to Real-World, Product to Real-World

- Baseline: use an SVM with rbf kernel, with deep learning features

  | SVM RBF   | Art   | RealWorld |       |       |       |       |       |       |       |       |
  | --------- | ----- | --------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
  | C         | 0.1   | 0.2       | 0.3   | 0.4   | 0.5   | 0.6   | 0.7   | 0.8   | 0.9   | 1     |
  | train acc | 43.9% | 67.7%     | 79.7% | 87.5% | 91.7% | 94.4% | 96.3% | 97.2% | 98.1% | 98.5% |
  | test acc  | 27.2% | 46.2%     | 57.8% | 64.1% | 67.5% | 69.9% | 71.7% | 73.1% | 73.9% | 74.4% |

  | SVM RBF   | Clipart | RealWorld |       |       |       |       |       |       |       |       |
  | --------- | ------- | --------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
  | C         | 0.1     | 0.2       | 0.3   | 0.4   | 0.5   | 0.6   | 0.7   | 0.8   | 0.9   | 1     |
  | train acc | 69.1%   | 87.9%     | 92.3% | 93.8% | 95.1% | 96.0% | 96.5% | 97.0% | 97.3% | 97.5% |
  | test acc  | 41.7%   | 56.0%     | 59.8% | 61.4% | 62.6% | 63.3% | 63.9% | 64.3% | 64.6% | 64.9% |

  | SVM RBF   | Product | RealWorld |       |       |       |       |       |       |       |       |
  | --------- | ------- | --------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
  | C         | 0.1     | 0.2       | 0.3   | 0.4   | 0.5   | 0.6   | 0.7   | 0.8   | 0.9   | 1     |
  | train acc | 87.4%   | 96.1%     | 97.7% | 98.1% | 98.4% | 98.7% | 98.9% | 99.0% | 99.1% | 99.6% |
  | test acc  | 61.2%   | 70.3%     | 71.7% | 72.3% | 72.8% | 72.8% | 73.1% | 73.0% | 73.0% | 73.0% |

- Domain adaptation methods: 

  - CORAL

    | SVM RBF     | Art   | RealWorld |       |       |       |       |       |       |       |       |
    | ----------- | ----- | --------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
    | C           | 0.1   | 0.2       | 0.3   | 0.4   | 0.5   | 0.6   | 0.7   | 0.8   | 0.9   | 1     |
    | train acc   | 40.5% | 67.7%     | 79.8% | 86.4% | 90.6% | 93.6% | 95.1% | 96.6% | 97.4% | 98.1% |
    | test acc    | 26.6% | 46.6%     | 58.3% | 64.6% | 67.2% | 69.5% | 71.3% | 72.7% | 73.6% | 73.9% |
    | baseline    | 27.2% | 46.2%     | 57.8% | 64.1% | 67.5% | 69.9% | 71.7% | 73.1% | 73.9% | 74.4% |
    | improvement | -0.6% | 0.4%      | 0.5%  | 0.5%  | -0.3% | -0.4% | -0.4% | -0.4% | -0.3% | -0.5% |

    | SVM RBF     | Clipart | RealWorld |       |       |       |       |       |       |       |       |
    | ----------- | ------- | --------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
    | C           | 0.1     | 0.2       | 0.3   | 0.4   | 0.5   | 0.6   | 0.7   | 0.8   | 0.9   | 1     |
    | train acc   | 64.9%   | 86.3%     | 91.4% | 93.7% | 95.1% | 95.9% | 96.5% | 97.0% | 97.3% | 97.5% |
    | test acc    | 42.7%   | 57.2%     | 60.7% | 62.6% | 63.8% | 64.1% | 64.2% | 64.6% | 65.0% | 65.1% |
    | baseline    | 41.7%   | 56.0%     | 59.8% | 61.4% | 62.6% | 63.3% | 63.9% | 64.3% | 64.6% | 64.9% |
    | improvement | 1.0%    | 1.2%      | 0.9%  | 1.2%  | 1.2%  | 0.8%  | 0.3%  | 0.3%  | 0.4%  | 0.2%  |

    | SVM RBF     | Product | RealWorld |       |       |       |       |       |       |       |       |
    | ----------- | ------- | --------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
    | C           | 0.1     | 0.2       | 0.3   | 0.4   | 0.5   | 0.6   | 0.7   | 0.8   | 0.9   | 1     |
    | train acc   | 85.2%   | 95.8%     | 97.5% | 98.1% | 98.5% | 98.7% | 98.9% | 99.1% | 99.2% | 99.3% |
    | test acc    | 60.1%   | 70.5%     | 72.0% | 72.5% | 72.8% | 72.7% | 72.8% | 73.0% | 73.0% | 73.1% |
    | baseline    | 61.2%   | 70.3%     | 71.7% | 72.3% | 72.8% | 72.8% | 73.1% | 73.0% | 73.0% | 73.0% |
    | improvement | -1.1%   | 0.2%      | 0.3%  | 0.2%  | 0.0%  | -0.1% | -0.3% | 0.0%  | 0.0%  | 0.1%  |

  - BDA

    |                   | dimension | 32     | 64     | 128    | 256    | 512    | 1024   | 2048   | baseline |
    | ----------------- | --------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | -------- |
    | Art_RealWorld     | train     | 89.3%  | 92.2%  | 94.3%  | 94.4%  | 94.9%  | 95.1%  | 95.1%  | 98.5%    |
    |                   | test      | 70.3%  | 74.1%  | 74.8%  | 74.3%  | 74.2%  | 74.2%  | 74.2%  | 74.4%    |
    | Clipart_RealWorld | train     | 93.0%  | 95.0%  | 95.4%  | 95.4%  | 95.6%  | 95.7%  | 95.8%  | 97.5%    |
    |                   | test      | 63.6%  | 65.3%  | 65.7%  | 65.7%  | 65.5%  | 65.5%  | 65.4%  | 64.9%    |
    | Product_RealWorld | train     | 97.40% | 98.20% | 98.50% | 98.50% | 98.60% | 98.80% | 98.80% | 99.6%    |
    |                   | test      | 71.70% | 73.80% | 73.50% | 74.10% | 74.20% | 74.20% | 74.20% | 73.0%    |
  
  - Deep CORAL
  
    | resnet50 |                  |                  |                  |          |
    | -------- | ---------------- | ---------------- | ---------------- | -------- |
    | $\eta$   | $10^3$ to $10^3$ | $10^3$ to $10^4$ | $10^3$ to $10^5$ | baseline |
    | A$\to$R  | 72.29%           | 71.26%           | 72.04%           | 74.4%    |
    | C$\to$R  | 62.29%           | 62.47%           | 62.47%           | 64.9%    |
    | P$\to$R  | 71.01%           | 71.52%           | 71.56%           | 73.0%    |
  
    | backbone | AlexNet | ResNet-18 | ResNet-34 | ResNet-50 | ResNet-101 | baseline |
    | -------- | ------- | --------- | --------- | --------- | ---------- | -------- |
    | A$\to$R  | 43.92%  | 66.90%    | 70.66%    | 71.26%    | 74.13%     | 74.4%    |
    | C$\to$R  | 36.56%  | 56.09%    | 59.67%    | 62.47%    | 64.14%     | 64.9%    |
    | P$\to$R  | 44.22%  | 65.29%    | 69.54%    | 71.52%    | 73.62%     | 73.0%    |
