import os
import os.path as osp
import numpy as np

feature_root = "./data/Office-Home_resnet50"
for file in os.listdir(feature_root):
    if file.startswith("readme"):
        continue
    with open(osp.join(feature_root, file), "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].split(",")
            lines[i] = [float(elem) for elem in lines[i]]
    lines = np.array(lines)
    features = lines[:, :-1]
    labels = lines[:, -1].astype(np.int64)
    print()
    print("Info of file:", file)
    print(" |- number of samples:", len(lines))
    print(" |- line length:", len(lines[0]), len(lines[-1]))
    print(" |- feature shape:", features.shape, features.dtype)
    print(" |- label shape:", labels.shape, labels.dtype)

