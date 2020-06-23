import os
import os.path as osp

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from utils import makedirs

def resize(data_root="./data/OfficeHomeDataset", resize=256):
    save_folder = "./data/OfficeHomeDataset_{}".format(resize)
    makedirs(save_folder)
    for domain in os.listdir(data_root):
        if domain.endswith('txt') or domain.endswith('csv'):
            continue
        domain_folder = osp.join(data_root, domain)
        domain_save_folder = osp.join(save_folder, domain)
        for category in tqdm(os.listdir(domain_folder), mininterval=1, desc=f"{domain}"):
            for filename in os.listdir(osp.join(domain_folder, category)):
                makedirs(osp.join(domain_save_folder, category))
                image = Image.open(osp.join(domain_folder, category, filename))
                image = image.resize((resize, resize))
                image.save(osp.join(domain_save_folder, category, filename))



resize(resize=256)