#!/usr/bin/python3
# -*- coding: utf-8 -*
from torch.utils.data import Dataset
from PIL import Image
from utils.tools import make_dataset, process_multiple_mask_tensor

# DATA_PATH = '/home/tangwenqi/workspace/data/small_g4'
BASE_PATH = '/home/tangwenqi/workspace/data/small_g4_all'

# COLOR_MAP = {'background': 0, 'gn': 63, 'gs': 127, 'gm': 191, 'gl': 255}
COLOR_MAP = {
    0: 0,
    63: 1,
    127: 2,
    191: 3,
    255: 4
}


# CLASS_WEIGHT = [8.83903352e-06, 1.00000000e+00, 3.36325895e+00, 1.03720893e+00, 5.85139039e-01]

# # 将所有图像转换为灰度图像（单channel）后计算的参数
# MEAN = [0.36667065542216914]
# STD = [0.18599752868073624]
# CHANNELS_NUM = 1


class Glomerulus(Dataset):
    """coco_train肾小球数据集，5类（包含背景），总共937张图像，训练集、验证集73分，原图尺寸1360*1024"""

    CHANNELS_NUM = 3
    NUM_CLASSES = 5

    MEAN = [0.39150069, 0.34523284, 0.43060046]
    STD = [0.1885685, 0.1950442, 0.18506475]

    def __init__(self, mode, transform=None, target_transform=None):
        self.items = make_dataset(mode, BASE_PATH, is_contain_augmented_data=True)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.items)

    def __str__(self):
        return 'Glomerulus'

    def __getitem__(self, index):
        image_path = self.items[index]['image_path']
        mask_path = self.items[index]['mask_path']

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        # 对tensor类型的mask进行处理
        mask = process_multiple_mask_tensor(mask, COLOR_MAP)

        return image, mask
