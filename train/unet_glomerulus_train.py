#!/usr/bin/python3
# -*- coding: utf-8 -*
import sys

sys.path.append('..')

import torch
import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.unet import UNet
from datasets import glomerulus
from torchvision import transforms
import os
import argparse
import numpy as np
from utils.metrics import compute_metrics
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', choices=['0', '1'])
parser.add_argument('--epoch', type=int, default=1000)
# TODO: 记得修改batch
parser.add_argument('--batch_size', type=int, default=80)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--parallel', type=bool, default=True)
parser.add_argument('--print_frequency', type=int, default=3)
parser.add_argument('--save_frequency', type=int, default=100)
args = parser.parse_args()

ROOT_PATH = '/home/tangwenqi/workspace/pycharm_workspace/Semantic-Segmentation-of-Pathological-Images/'
CHECKPOINT_DIR_PATH = os.path.join(ROOT_PATH, 'checkpoint')
# TODO: 记得修改寸尺tensorboard log的路径
LOGS_PATH = os.path.join(ROOT_PATH, 'log', 'unet_glomerulus')

# 创建Tensorboard可视化的log目录，若存在，则删除再创建
if os.path.exists(LOGS_PATH):
    shutil.rmtree(LOGS_PATH)
os.mkdir(LOGS_PATH)
writer = SummaryWriter(log_dir=LOGS_PATH, flush_secs=60)

DEVICE = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

# 超参数
EPOCH = args.epoch
BATCH_SIZE = args.batch_size
LR = args.lr
WEIGHT_DECAY = args.weight_decay
NUM_CLASSES = glomerulus.NUM_CLASSES

print('<================== Parameters ==================>')
print('model: UNet')
print('batch_size: {}'.format(BATCH_SIZE))
print('lr: {}'.format(LR))
print('weight_decay: {}'.format(WEIGHT_DECAY))
print('<================================================>')

# 加载数据
print('Loading data...')

# 对image和mask进行resize
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(),
                                transforms.Normalize(mean=glomerulus.MEAN,
                                                     std=glomerulus.STD)])
target_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

train_data = glomerulus.Glomerulus(mode='train', transform=transform, target_transform=target_transform)
val_data = glomerulus.Glomerulus(mode='val', transform=transform, target_transform=target_transform)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)

# 初始化模型、优化方法、损失函数
# TODO: 记得修改模型
net = UNet(num_classes=NUM_CLASSES)
# 判断是否使用多GPU运行
if args.parallel:
    print('Use DataParallel.')
    net = torch.nn.DataParallel(net)
net = net.to(DEVICE)
# TODO: 下次运行需要调参了，可以引入学习率变化的一些策略
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# Pytorch交叉熵能够用于2D数据
criterion = torch.nn.CrossEntropyLoss()

temp = 0
print('Start training...')
for epoch in range(args.epoch):
    # 训练
    loss_all = []
    predictions_all = []
    labels_all = []

    net.train()
    for index, data in enumerate(train_loader):
        inputs, labels = data
        # label读取进来转为tensor后，shape为B*1*H*W，去掉dim=1
        labels = torch.squeeze(labels, dim=1)
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = net(inputs)
        # 计算在该批次上的平均损失函数
        loss = criterion(outputs, labels.long()) / inputs.size(0)
        loss.backward()
        optimizer.step()

        loss_all.append(loss.item())
        # 将概率最大的类别作为预测的类别
        predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int)
        labels = labels.cpu().numpy().astype(np.int)

        predictions_all.append(predictions)
        labels_all.append(labels)

        if (index + 1) % args.print_frequency == 0:
            # 计算打印间隔的平均损失函数
            avg_loss = np.mean(loss_all)
            loss_all = []

            writer.add_scalar('train/loss', avg_loss, temp)
            temp += 1

            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} ".format(
                epoch + 1, EPOCH, index + 1, len(train_loader), avg_loss))

    # 使用混淆矩阵计算语义分割中的指标
    pa, mpa, miou, fwiou = compute_metrics(predictions_all, labels_all, NUM_CLASSES)

    writer.add_scalars('train/metrics', dict(pa=pa, mpa=mpa, miou=miou, fwiou=fwiou), epoch)

    print('Training: PA: {:.4f}, MPA: {:.4f}, MIoU: {:.4f}, FWIoU: {:.4f}'.format(pa, mpa, miou, fwiou))

    # 验证
    loss_all = []
    predictions_all = []
    labels_all = []

    net.eval()
    with torch.no_grad():
        for _, data in enumerate(val_loader):
            inputs, labels = data
            # label读取进来转为tensor后，shape为B*1*H*W，去掉dim=1
            labels = torch.squeeze(labels, dim=1)
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = net(inputs)
            loss = criterion(outputs, labels.long()) / inputs.size(0)

            loss_all.append(loss.item())
            # 将概率最大的类别作为预测的类别
            predictions = torch.max(outputs, dim=1)[1].cpu().numpy().astype(np.int)
            labels = labels.cpu().numpy().astype(np.int)

            predictions_all.append(predictions)
            labels_all.append(labels)

    # 使用混淆矩阵计算语义分割中的指标
    pa, mpa, miou, fwiou = compute_metrics(predictions_all, labels_all, NUM_CLASSES)
    avg_loss = np.mean(loss_all)

    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalars('val/metrics', dict(pa=pa, mpa=mpa, miou=miou, fwiou=fwiou), epoch)

    print('Validation: PA: {:.4f}, MPA: {:.4f}, MIoU: {:.4f}, FWIoU: {:.4f}, Loss: {:.4f}'.
          format(pa, mpa, miou, fwiou, avg_loss))

    # 保存模型参数和优化器参数
    if (epoch + 1) % args.save_frequency == 0:
        # TODO: 记得修改输出输出模型的名字前缀
        checkpoint_path = os.path.join(CHECKPOINT_DIR_PATH,
                                       'unet_glomerulus_{}.pkl'.format(
                                           time.strftime('%m%d_%H%M', time.localtime())))
        torch.save({
            'is_parallel': args.parallel,
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
        print('Save model at {}.'.format(checkpoint_path))
