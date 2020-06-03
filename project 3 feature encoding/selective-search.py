import selectivesearch
import cv2
import argparse
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import net
from utils import get_awa2_split_paths, makedirs

AWA2_mean = [0.485, 0.456, 0.406]
AWA2_std = [0.229, 0.224, 0.225]

class SSFeatureExtractor(object):
    def __init__(self, args):
        self.pretrained_mode = args.pretrained_mode
        self.arch = args.arch
        self.gpu_id = args.gpu_id
        self.data_root = args.data_root
        self.scale = args.scale
        self.sigma = args.sigma
        self.min_size = args.min_size
        self.proposal_mode = args.proposal_mode
        self.save_root = f"./save-ss/{self.arch}_{self.pretrained_mode}_{self.proposal_mode}"
        makedirs(self.save_root)

        self._prepare_dataset()
        self._prepare_model()

    def _prepare_dataset(self):
        self.PIL2Tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(AWA2_mean, AWA2_std)
        ])
        self.all_classes, self.all_images = get_awa2_split_paths(self.data_root)

    def _prepare_model(self):
        if self.pretrained_mode == 'awa2':
            num_classes = 50
            self.model = net.__dict__[self.arch](num_classes=num_classes, output_feature=True).to(self.gpu_id)
            self.model.load_state_dict(torch.load(f'./save-pretrained/{self.arch}/model.pth'))
        elif self.pretrained_mode == 'imagenet':
            num_classes = 1000
            self.model = net.__dict__[self.arch](num_classes=num_classes, pretrained=True, output_feature=True).to(self.gpu_id)
        else:
            raise Exception("pretrained mode {} not supported".format(self.pretrained_mode))
        self.model.eval()

    def _extract_feature(self, image_path, vis_save_path, des_save_path):
        # Step 1: Extract proposals
        image = cv2.imread(image_path)
        img_lbl, regions = selectivesearch.selective_search(image, scale=self.scale, sigma=self.sigma,
                                                            min_size=self.min_size)  # (left, top, width, height)
        proposals = []
        for region in regions:
            x, y, w, h = region['rect']
            if w == 0 or h == 0:
                continue
            proposals.append((x, y, x+w, y+h))
        # Step 2: Visualize proposals
        image_annot = image
        for x1, y1, x2, y2 in proposals:
            image_annot = cv2.rectangle(image_annot, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(vis_save_path, image_annot)
        # Step 3: Generate batch using the proposals
        try:
            image_torch = self.PIL2Tensor(Image.open(image_path)).to(self.gpu_id)
        except:
            image_torch = self.PIL2Tensor(Image.open(image_path).convert("RGB")).to(self.gpu_id)
        batch = self._generate_batch(image_torch, proposals)
        # Step 4: Calculate features
        with torch.no_grad():
            features = self.model(batch)
            features = features.cpu().numpy()
        np.save(des_save_path, features)

    def _generate_batch(self, image_torch, proposals):
        batch = None
        if self.proposal_mode == "topleft":
            for x1, y1, x2, y2 in proposals:
                region = image_torch[:, x1:x2, y1:y2]
                image = torch.zeros(3, 224, 224).to(self.gpu_id)
                image[:, 0:x2-x1, 0:y2-y1] = region
                if batch is None:
                    batch = image.unsqueeze(0)
                else:
                    batch = torch.cat((batch, image.unsqueeze(0)), dim=0)
            return batch
        elif self.proposal_mode == "original":
            for x1, y1, x2, y2 in proposals:
                region = image_torch[:, x1:x2, y1:y2]
                image = torch.zeros(3, 224, 224).to(self.gpu_id)
                image[:, x1:x2, y1:y2] = region
                if batch is None:
                    batch = image.unsqueeze(0)
                else:
                    batch = torch.cat((batch, image.unsqueeze(0)), dim=0)
            return batch
        elif self.proposal_mode == "resize":
            image_torch = image_torch.cpu()
            resize = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            for x1, y1, x2, y2 in proposals:
                region = image_torch[:, x1:x2, y1:y2]
                image = resize(region).to(self.gpu_id)
                if batch is None:
                    batch = image.unsqueeze(0)
                else:
                    batch = torch.cat((batch, image.unsqueeze(0)), dim=0)
            return batch
        else:
            raise Exception(f"Proposal mode {self.proposal_mode} not supported.")

    def run(self):
        for label, image_paths in self.all_images['train'].items():
            vis_save_folder = osp.join(self.save_root, "visualization", "train", label)
            des_save_folder = osp.join(self.save_root, "descriptor", "train", label)
            makedirs(vis_save_folder)
            makedirs(des_save_folder)
            for image_path in tqdm(image_paths, desc=label + " (train)", mininterval=1):
                image_name = image_path.split("/")[-1]
                image_name = image_name.split(".")[0]
                vis_save_path = osp.join(vis_save_folder, f"{image_name}.png")
                des_save_path = osp.join(des_save_folder, f"{image_name}.npy")
                self._extract_feature(image_path, vis_save_path, des_save_path)

        for label, image_paths in self.all_images['test'].items():
            vis_save_folder = osp.join(self.save_root, "visualization", "test", label)
            des_save_folder = osp.join(self.save_root, "descriptor", "test", label)
            makedirs(vis_save_folder)
            makedirs(des_save_folder)
            for image_path in tqdm(image_paths, desc=label + " (test)", mininterval=1):
                image_name = image_path.split("/")[-1]
                image_name = image_name.split(".")[0]
                vis_save_path = osp.join(vis_save_folder, f"{image_name}.png")
                des_save_path = osp.join(des_save_folder, f"{image_name}.npy")
                self._extract_feature(image_path, vis_save_path, des_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Selective Search")
    parser.add_argument('--scale', type=int, default=200)
    parser.add_argument('--sigma', type=float, default=0.8)
    parser.add_argument('--min-size', type=int, default=200)
    parser.add_argument('--data-root', type=str, default='./data/split_40_resize')
    parser.add_argument('--proposal-mode', type=str, default='resize', help='topleft, original, resize')
    parser.add_argument('--pretrained-mode', type=str, default='awa2', help='awa2, imagenet')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--gpu-id', type=int, default=0)
    args = parser.parse_args()

    extractor = SSFeatureExtractor(args)
    extractor.run()