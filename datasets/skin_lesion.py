#!/usr/bin/python3
# -*- coding: utf-8 -*
from torch.utils.data import Dataset
from PIL import Image
from utils.tools import make_dataset, process_binary_mask_tensor

BASE_PATH = '/home/tangwenqi/workspace/data/Skin-Lesion-Segmentation-ISIC-2017'


class SkinLesion(Dataset):
    """肾小球数据集，2类（包含背景），总共937张图像，训练集、验证集73分，原图尺寸1360*1024"""

    CHANNELS_NUM = 3
    NUM_CLASSES = 2

    MEAN = [0.70791537, 0.59156666, 0.54687498]
    STD = [0.15324752, 0.16178547, 0.17681521]

    def __init__(self, mode, transform=None, target_transform=None):
        self.items = make_dataset(mode, BASE_PATH, is_contain_augmented_data=False)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.items)

    def __str__(self):
        return 'SkinLesion'

    def __getitem__(self, index):
        image_path = self.items[index]['image_path']
        mask_path = self.items[index]['mask_path']

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        mask = process_binary_mask_tensor(mask)

        return image, mask
