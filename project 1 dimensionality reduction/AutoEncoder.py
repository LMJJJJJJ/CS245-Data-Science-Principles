import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim

import os
import os.path as osp
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


class AutoEncoder(nn.Module):
    def __init__(self, n_components):
        super(AutoEncoder, self).__init__()
        self.n_components = n_components
        self.encoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_components)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_components, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048)
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out

    def encode(self, x):
        return self.encoder(x)


class Trainer(object):
    def __init__(self, args):
        self.N_COMPONENTS = args.n_components
        self.GPU_ID = args.gpu_id
        self.BATCH_SIZE = args.batch_size
        self.INITIAL_LR = args.initial_lr
        self.FINAL_LR = args.final_lr
        self.NUM_EPOCHS = args.num_epochs
        self.LR_LIST = np.logspace(self.INITIAL_LR, self.FINAL_LR, self.NUM_EPOCHS)
        self.DATA_ROOT = args.data_root
        self.NUM_WORKERS = args.num_workers

        self.PATH = {}
        self.PATH["model_save_folder"] = './AE-train-save/models'
        self.PATH["fig_save_folder"] = "./AE-train-save/figs"
        self.PATH["data_save_folder"] = "./AE-train-save/data"
        name = f"{self.N_COMPONENTS}_{self.INITIAL_LR}_{self.FINAL_LR}_{self.NUM_EPOCHS}"
        self.PATH["model_save_path"] = osp.join(self.PATH["model_save_folder"], f"{name}.pth")
        self.PATH["data_save_path"] = osp.join(self.PATH["data_save_folder"], f"{name}.bin")
        self.PATH["fig_save_path"] = osp.join(self.PATH["fig_save_folder"], f"{name}.png")

        self.PATH["encoded_save_folder"] = osp.join(f"./data/AE/{name}")

        if not os.path.exists(self.PATH["model_save_folder"]):
            os.makedirs(self.PATH["model_save_folder"])
        if not os.path.exists(self.PATH["fig_save_folder"]):
            os.makedirs(self.PATH["fig_save_folder"])
        if not os.path.exists(self.PATH["data_save_folder"]):
            os.makedirs(self.PATH["data_save_folder"])
        if not os.path.exists(self.PATH["encoded_save_folder"]):
            os.makedirs(self.PATH["encoded_save_folder"])

        self._prepare()

    def _prepare(self):
        self._prepare_plot_dic()
        self._prepare_model()
        self._prepare_dataset()

    def _prepare_plot_dic(self):
        self.plot_dic = {
            'loss': []
        }

    def _prepare_dataset(self):
        X = np.load(osp.join(self.DATA_ROOT, 'feature.npy'))
        y = np.load(osp.join(self.DATA_ROOT, 'label.npy'))
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.BATCH_SIZE,
            shuffle=True, num_workers=self.NUM_WORKERS
        )

    def _prepare_model(self):
        self.model = AutoEncoder(self.N_COMPONENTS).to(self.GPU_ID)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR_LIST[0], betas=(0.9, 0.999),
                                          eps=1e-08, weight_decay=0, amsgrad=False)
        self.criterion = nn.MSELoss()

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        self.update_lr(self.LR_LIST[self.EPOCH - 1])
        print("Learning rate is", self.optimizer.param_groups[0]["lr"])
        for features, _ in tqdm(self.data_loader, desc="epoch " + str(self.EPOCH), mininterval=1):
            features = features.to(self.GPU_ID)
            features_hat = self.model(features)
            self.optimizer.zero_grad()
            loss = self.criterion(features_hat, features)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            train_loss += loss.data * features.shape[0]
        train_loss = train_loss / len(self.data_loader.dataset)
        self.plot_dic['loss'].append(train_loss.item())
        print("Train loss: {0:.5f}".format(train_loss))

    def update_lr(self, lr):
        for ix, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr

    def draw(self):
        print("Plotting...")
        plt.figure(figsize=(8, 6))
        # train & test accuracy
        plt.subplot(1, 1, 1)
        x = range(1, len(self.plot_dic["loss"]) + 1)
        plt.xlabel("epoch")
        plt.plot(x, self.plot_dic["loss"], label="loss")
        plt.legend()
        # PLOT
        plt.tight_layout()
        plt.savefig(self.PATH['fig_save_path'], bbox_inches='tight', dpi=300)
        plt.close("all")

    def save(self):
        print("Saving...")
        save_obj(self.plot_dic, self.PATH['data_save_path'])
        torch.save(self.model.state_dict(), self.PATH['model_save_path'])

    def save_encoded(self):
        self.model.eval()
        X = y = None
        with torch.no_grad():
            for features, labels in tqdm(self.data_loader, desc="saving", mininterval=1):
                features = features.to(self.GPU_ID)
                features = self.model.encode(features).cpu()
                if X is None:
                    X = features
                    y = labels
                else:
                    X = torch.cat((X, features), dim=0)
                    y = torch.cat((y, labels), dim=0)
        X = X.numpy()
        y = y.numpy()
        np.save(osp.join(self.PATH["encoded_save_folder"], "feature.npy"), X)
        np.save(osp.join(self.PATH["encoded_save_folder"], "label.npy"), y)

    def run(self):
        for self.EPOCH in range(1, self.NUM_EPOCHS + 1):
            self.train_epoch()
            self.draw()
            self.save()
        torch.save(self.model.cpu().state_dict(), self.PATH['model_save_path'])
        self.model.to(self.GPU_ID)
        self.save_encoded()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AutoEncoder")
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--initial-lr', type=int, default=-3)
    parser.add_argument('--final-lr', type=int, default=-4)
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--data-root', type=str, default='./data/raw')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--n-components', type=int, default=64)
    args = parser.parse_args()
    print(args)

    trainer = Trainer(args)
    trainer.run()