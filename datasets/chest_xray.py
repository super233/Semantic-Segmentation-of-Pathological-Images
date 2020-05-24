#!/usr/bin/python3
# -*- coding: utf-8 -*
from torch.utils.data import Dataset
from PIL import Image
from utils.tools import make_dataset, process_binary_mask_tensor

BASE_PATH = '/home/tangwenqi/workspace/data/Chest-Xray-Masks-and-Labels'


class ChestXray(Dataset):
    """肺部的X光数据，带有肺叶的标注"""

    CHANNELS_NUM = 3
    NUM_CLASSES = 2

    MEAN = [0.58479634, 0.58479634, 0.58479634]
    STD = [0.27773262, 0.27773262, 0.27773262]

    def __init__(self, mode, transform=None, target_transform=None):
        self.items = make_dataset(mode, BASE_PATH, is_contain_augmented_data=False)
        self.transform = transform
        self.target_transform = target_transform

    def __str__(self):
        return 'ChestXray'

    def __len__(self):
        return len(self.items)

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
