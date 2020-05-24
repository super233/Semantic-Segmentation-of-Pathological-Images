#!/usr/bin/python3
# -*- coding: utf-8 -*
from torch.utils.data import Dataset
from PIL import Image
import pydicom
from utils.tools import make_dataset, process_multiple_mask_tensor
import os
import random

BASE_PATH = '/home/tangwenqi/workspace/data/CHAOS'
COLOR_MAP = {
    63: 1,
    126: 2,
    189: 3,
    252: 4
}


class NoisyChaos(Dataset):
    """
    用于从MR-T2SPIR图像中分割肝、脾、右肾、左肾的数据集，共20个文件夹,
    原图是.dcm格式的，读取进来数据类型为uint16
    """

    # 转为RGB格式
    CHANNELS_NUM = 3
    # 加上背景共5类
    NUM_CLASSES = 5

    # 所有图像3个channel上的normalization参数
    MEAN = [0.25229187, 0.25229187, 0.25229187]
    STD = [0.37067029, 0.37067029, 0.37067029]

    def __init__(self, mode, transform=None, target_transform=None, noisy_rate=0.2, noisy_type='sy'):
        self.items = make_dataset(mode, BASE_PATH, is_contain_augmented_data=False)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        if mode == 'train':
            assert noisy_type in ['sy', 'asy']

            self.noisy_rate = noisy_rate
            self.noisy_type = noisy_type
            self.noisy_indexes = self.generate_noisy_indexes()

    def __str__(self):
        return 'CHAOS(noisy_rate={}, noisy_type={})'.format(self.noisy_rate, self.noisy_type)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_path = self.items[index]['image_path']
        mask_path = self.items[index]['mask_path']

        # 使用噪声图像
        if self.mode == 'train' and index in self.noisy_indexes:
            mask_path = '{}_noisy_{}.png'.format(mask_path.split('.png')[0], self.noisy_type)

        image = Image.fromarray(pydicom.read_file(image_path).pixel_array).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        # 对tensor类型的mask进行处理
        mask = process_multiple_mask_tensor(mask, COLOR_MAP)

        return image, mask

    def generate_noisy_indexes(self, seed=0):
        """随机选取指定数量的噪声图像，获取其index"""

        noisy_indexes = []
        # 获得所有含有噪声图像的index
        for index in range(len(self.items)):
            noisy_mask_path = '{}_noisy_{}.png'.format(self.items[index]['mask_path'].split('.png')[0], self.noisy_type)
            if os.path.exists(noisy_mask_path):
                noisy_indexes.append(index)

        random.seed(seed)
        random.shuffle(noisy_indexes)

        select_num = int(self.__len__() * self.noisy_rate)

        return noisy_indexes[:select_num]
