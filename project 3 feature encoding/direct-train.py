import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
import os.path as osp
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import net
from utils import makedirs, save_obj

AWA2_mean = [0.485, 0.456, 0.406]
AWA2_std = [0.229, 0.224, 0.225]
AWA2_classes = 50

class Trainer(object):
    def __init__(self, args):
        self.data_root = args.data_root
        self.batch_size = args.batch_size
        self.arch = args.arch
        self.gpu_id = args.gpu_id
        self.num_epochs = args.num_epochs
        self.initial_lr = args.initial_lr
        self.final_lr = args.final_lr
        self.lr_list = np.logspace(self.initial_lr, self.final_lr, self.num_epochs)

        self.PATH = {}
        self.PATH["save_root"] = "./save-pretrained/{}".format(self.arch)
        self.PATH["model_save_path"] = osp.join(self.PATH["save_root"], "model.pth")
        self.PATH["fig_save_path"] = osp.join(self.PATH["save_root"], "fig.png")
        self.PATH["data_save_path"] = osp.join(self.PATH["save_root"], "data.bin")
        makedirs(self.PATH["save_root"])
        self.prepare()

    def prepare(self):
        self._prepare_plot_dict()
        self._prepare_dataset()
        self._prepare_model()

    def _prepare_plot_dict(self):
        self.plot_dic = {
            "train_acc": [],
            "test_acc": [],
            "loss": []
        }

    def _prepare_dataset(self):
        print("# Data prepared, root:", self.data_root)
        train_transform = transforms.Compose([
            transforms.RandomCrop(224, 8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(AWA2_mean, AWA2_std)
        ])
        train_set = datasets.ImageFolder(osp.join(self.data_root, "train"), train_transform)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(AWA2_mean, AWA2_std)
        ])
        test_set = datasets.ImageFolder(osp.join(self.data_root, "test"), test_transform)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

    def _prepare_model(self):
        self.model = net.__dict__[self.arch](num_classes=AWA2_classes).to(self.gpu_id)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999),
                                          eps=1e-08, weight_decay=0, amsgrad=False) # lr is default.
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.train()
        train_loss = 0
        correct = 0
        self.update_lr(self.lr_list[self.EPOCH - 1])
        print("Learning rate is", self.optimizer.param_groups[0]["lr"])
        for images, labels in tqdm(self.train_loader, desc="epoch " + str(self.EPOCH), mininterval=1):
            images, labels = images.to(self.gpu_id), labels.to(self.gpu_id)
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.criterion(output, labels)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            train_loss += loss.data * images.shape[0]
            y_pred = output.data.max(1)[1]
            correct += y_pred.eq(labels.data).sum()

        acc = 100. * float(correct) / len(self.train_loader.dataset)
        train_loss = train_loss / len(self.train_loader.dataset)
        self.plot_dic["train_acc"].append(acc)
        self.plot_dic["loss"].append(train_loss.item())
        print(" -- Epoch {}: loss {}, acc(train) {:.2f}".format(self.EPOCH, train_loss, acc))

    def validate(self):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="evaluation", mininterval=1):
                images, labels = images.to(self.gpu_id), labels.to(self.gpu_id)
                output = self.model(images)
                prediction = output.data.max(1)[1]
                correct += prediction.eq(labels.data).sum()
        acc = 100. * float(correct) / len(self.test_loader.dataset)
        self.plot_dic["test_acc"].append(acc)
        print(" -- Epoch {}: acc(val) {:.2f}".format(self.EPOCH, acc))

    def draw(self):
        print("Plotting...")
        plt.figure(figsize=(16, 12))
        # train & test accuracy
        plt.subplot(2, 2, 1)
        x = range(1, len(self.plot_dic["train_acc"]) + 1)
        plt.xlabel("epoch")
        plt.plot(x, self.plot_dic["train_acc"], label="train_acc")
        plt.plot(x, self.plot_dic["test_acc"], label="test_acc")
        plt.legend()
        # label loss (CrossEntropy) on training set
        plt.subplot(2, 2, 2)
        x = range(1, len(self.plot_dic["loss"]) + 1)
        plt.xlabel("epoch")
        plt.plot(x, self.plot_dic["loss"], label="loss")
        # PLOT
        plt.tight_layout()
        plt.savefig(self.PATH['fig_save_path'], bbox_inches='tight', dpi=300)
        plt.close("all")

    def save(self):
        print("# Saving model and data...")
        torch.save(self.model.state_dict(), self.PATH['model_save_path'])
        save_obj(self.plot_dic, self.PATH['data_save_path'])

    def update_lr(self, lr):
        for ix, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr

    def run(self):
        for self.EPOCH in range(1, self.num_epochs + 1):
            self.train()
            self.validate()
            self.draw()
            self.save()
        torch.save(self.model.to("cpu").state_dict(), self.PATH['model_save_path'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AwA2 Trainer (train directly)")
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--data-root', type=str, default='./data/split_40_resize')
    parser.add_argument('--arch', type=str, default='resnet34')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--initial-lr', type=int, default=-3)
    parser.add_argument('--final-lr', type=int, default=-5)
    parser.add_argument('--num-epochs', type=int, default=400)
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.run()