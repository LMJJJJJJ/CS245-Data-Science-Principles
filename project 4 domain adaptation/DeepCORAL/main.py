import torch
import os
import math
import data_loader
import models
from config import CFG
import utils
import numpy as np
from utils import save_obj, makedirs
import os.path as osp
import matplotlib.pyplot as plt
import argparse

log = []


plot_dic = {
    'cls_loss': [],
    'transfer_loss': [],
    'test_acc': []
}


def test(model, target_test_loader):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)

    print('{} --> {}: max correct: {}, accuracy{: .2f}%\n'.format(
        source_name, target_name, correct, 100. * correct / len_target_dataset))

    plot_dic['test_acc'].append(100. * correct / len_target_dataset)

    return 100. * correct / len_target_dataset


def train(source_loader, target_train_loader, target_test_loader, model, optimizer, CFG):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    save_folder = "./save/{}".format(CFG['backbone'])
    makedirs(save_folder)
    save_name = '{}_{}_{}_{}'.format(source_name, target_name, args.lambda_initial, args.lambda_final)
    for e in range(CFG['epoch']):
        print("lambda", CFG['lambda'][e])
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.train()
        iter_source, iter_target = iter(
            source_loader), iter(target_train_loader)
        n_batch = min(len_source_loader, len_target_loader)
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(n_batch):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            data_source, label_source = data_source.to(
                DEVICE), label_source.to(DEVICE)
            data_target = data_target.to(DEVICE)

            optimizer.zero_grad()
            label_source_pred, transfer_loss = model(data_source, data_target)
            clf_loss = criterion(label_source_pred, label_source)
            loss = clf_loss + CFG['lambda'][e] * transfer_loss
            loss.backward()
            optimizer.step()
            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
            if i % CFG['log_interval'] == 0:
                print(
                    'Train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.6f}, transfer_loss: {:.6f}, total_Loss: {:.6f}'.format(
                        e + 1,
                        CFG['epoch'],
                        int(100. * i / n_batch), train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg))
        plot_dic['cls_loss'].append(train_loss_clf.avg)
        plot_dic['transfer_loss'].append(train_loss_transfer.avg)
        # Test
        test_acc = test(model, target_test_loader)
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg, test_acc])
        np_log = np.array(log, dtype=float)
        np.savetxt(osp.join(save_folder, '{}.csv'.format(save_name)), np_log, delimiter=',',
                   fmt='%.6f')
        print("Saving", osp.join(save_folder, '{}.csv'.format(save_name)))
        plt.figure(figsize=(15, 4))
        # train & test accuracy
        plt.subplot(1, 3, 1)
        x = range(1, len(plot_dic['cls_loss']) + 1)
        plt.xlabel("epoch")
        plt.plot(x, plot_dic['cls_loss'], label="Classification Loss")
        plt.legend()
        # train & test loss
        plt.subplot(1, 3, 2)
        x = range(1, len(plot_dic['transfer_loss']) + 1)
        plt.xlabel("epoch")
        plt.plot(x, plot_dic['transfer_loss'], label="Generator Loss")
        plt.legend()
        #
        plt.subplot(1, 3, 3)
        x = range(1, len(plot_dic['test_acc']) + 1)
        plt.xlabel("epoch")
        plt.plot(x, plot_dic['test_acc'], label="Testing Accuracy")
        plt.legend()
        # PLOT
        plt.tight_layout()
        plt.savefig(osp.join(save_folder, '{}.png'.format(save_name)), bbox_inches='tight', dpi=300)
        plt.close("all")
        save_obj(plot_dic, osp.join(save_folder, '{}.bin'.format(save_name)))


def load_data(src, tar, root_dir):
    folder_src = root_dir + src
    folder_tar = root_dir + tar
    source_loader = data_loader.load_data(
        folder_src, CFG['batch_size'], True, CFG['kwargs'])
    target_train_loader = data_loader.load_data(
        folder_tar, CFG['batch_size'], True, CFG['kwargs'])
    target_test_loader = data_loader.load_data(
        folder_tar, CFG['batch_size'], False, CFG['kwargs'])
    return source_loader, target_train_loader, target_test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Deep Coral")
    parser.add_argument('--backbone', default='resnet50', help='alexnet, resnet18/34/50/101')
    parser.add_argument('--gpu-id', type=int, default=2)
    parser.add_argument('--source', default='Art')
    parser.add_argument('--lambda-initial', type=int, default=3)
    parser.add_argument('--lambda-final', type=int, default=4)
    args = parser.parse_args()
    print(args)
    DEVICE = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    CFG['backbone'] = args.backbone
    CFG['lambda'] = np.logspace(args.lambda_initial, args.lambda_final, CFG['epoch'])
    torch.manual_seed(0)

    source_name = args.source
    target_name = "RealWorld"

    print('Src: %s, Tar: %s' % (source_name, target_name))

    source_loader, target_train_loader, target_test_loader = load_data(
        source_name, target_name, CFG['data_path'])

    model = models.Transfer_Net(
        CFG['n_class'], transfer_loss='coral', base_net=CFG['backbone']).to(DEVICE)
    optimizer = torch.optim.SGD([
        {'params': model.base_network.parameters()},
        {'params': model.bottleneck_layer.parameters(), 'lr': 10 * CFG['lr']},
        {'params': model.classifier_layer.parameters(), 'lr': 10 * CFG['lr']},
    ], lr=CFG['lr'], momentum=CFG['momentum'], weight_decay=CFG['l2_decay'])

    train(source_loader, target_train_loader,
          target_test_loader, model, optimizer, CFG)