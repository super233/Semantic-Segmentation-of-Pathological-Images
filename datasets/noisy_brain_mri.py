#!/usr/bin/python3
# -*- coding: utf-8 -*
from torch.utils.data import Dataset
from PIL import Image
from utils.tools import make_dataset, process_binary_mask_tensor, generate_noisy_indexes

BASE_PATH = '/home/tangwenqi/workspace/data/Brain-MRI-segmentation'


class NoisyBrainMri(Dataset):
    """含有噪声的Brain MRI Segmentation数据，用于FLAIR异常的区域的分割"""

    CHANNELS_NUM = 3
    # 只有2类，FLAIR异常的区域和背景
    NUM_CLASSES = 2
    # 所有图像3个channel上的normalization参数
    MEAN = [0.09187034, 0.08331693, 0.08746525]
    STD = [0.13538943, 0.1237991, 0.12926382]

    def __init__(self, mode, transform=None, target_transform=None):
        self.items = make_dataset(mode, BASE_PATH, is_contain_augmented_data=False)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        # if mode == 'train':
        #     self.noisy_rate = 0.2
        #     self.noisy_indexes = generate_noisy_indexes(0, self.__len__(), self.noisy_rate)
        self.noisy_rate = 0.2
        self.noisy_indexes = generate_noisy_indexes(0, self.__len__(), self.noisy_rate)

    def __str__(self):
        return 'NoisyBrainMri(noisy_rate={})'.format(self.noisy_rate)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_path = self.items[index]['image_path']
        mask_path = self.items[index]['mask_path']

        # # 使用噪声图像
        # if self.mode == 'train' and index in self.noisy_indexes:
        #     mask_path = '{}_noisy.tif'.format(mask_path.split('.tif')[0])

        # 使用噪声图像
        if index in self.noisy_indexes:
            mask_path = '{}_noisy.tif'.format(mask_path.split('.tif')[0])

        # 原图为3通道RGB的格式，mask标注为单通道灰度图像格式
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        # 对tensor类型的mask进行处理
        mask = process_binary_mask_tensor(mask)

        return image, mask
