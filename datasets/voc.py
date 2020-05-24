#!/usr/bin/python3
# -*- coding: utf-8 -*
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image

NUM_CLASSES = 21
# png格式的标注数据，在mask边缘有255的白色区域
IGNORE_LABEL = 255


def make_dataset(mode):
    """加载训练集、测试集的数据和标签的路径"""

    assert mode in ['train', 'val']

    # item内存储(图像数据路径，标签数据路径)的元组
    items = []
    # 使用benchmark_RELEASE的数据作为训练集
    if mode == 'train':
        image_dir_path = '/home/tangwenqi/workspace/data/benchmark_RELEASE/dataset/img-transformed'
        mask_dir_path = '/home/tangwenqi/workspace/data/benchmark_RELEASE/dataset/cls-transformed'
        txt_path = '/home/tangwenqi/workspace/data/benchmark_RELEASE/dataset/train.txt'
    # 使用VOC2012的数据作为验证集
    else:
        image_dir_path = '/home/tangwenqi/workspace/data/VOCdevkit/VOC2012/JPEGImages-transformed'
        mask_dir_path = '/home/tangwenqi/workspace/data/VOCdevkit/VOC2012/SegmentationClass-transformed'
        txt_path = '/home/tangwenqi/workspace/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'

    with open(txt_path) as f:
        for l in f.readlines():
            file_name = l.strip('\n')
            item = (os.path.join(image_dir_path, file_name + '.jpg'),
                    os.path.join(mask_dir_path, file_name + '.png'))
            items.append(item)

    return items


class VOC(Dataset):
    """VOC数据集，共有21类"""

    def __init__(self, mode, transform=None, target_transform=None):
        super().__init__()
        # 根据mode获得对应的数据路径
        self.items = make_dataset(mode)
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_path, mask_path = self.items[index]
        # 图像读取时统一转换为3通道数据，避免其中奇奇怪怪的单通道数据
        image = Image.open(image_path).convert('RGB')

        mask = Image.open(mask_path)
        # 类型必须设置为np.int64，不然会被ToTensor()认为是图像被归一化到[0.0, 1.0]
        mask = np.asarray(mask, dtype=np.int64)
        # 这里必须清除掉这个多余的label，不然后面计算confusion matrix会报错
        mask[mask == IGNORE_LABEL] = 0

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        # 返回的mask是ndarray数据
        return image, mask
