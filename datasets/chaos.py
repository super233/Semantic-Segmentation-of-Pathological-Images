#!/usr/bin/python3
# -*- coding: utf-8 -*
from torch.utils.data import Dataset
from PIL import Image
import pydicom
from utils.tools import make_dataset, process_multiple_mask_tensor

BASE_PATH = '/home/tangwenqi/workspace/data/CHAOS'
COLOR_MAP = {
    63: 1,
    126: 2,
    189: 3,
    252: 4
}


class Chaos(Dataset):
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

    def __init__(self, mode, transform=None, target_transform=None):
        self.items = make_dataset(mode, BASE_PATH, is_contain_augmented_data=False)
        self.transform = transform
        self.target_transform = target_transform

    def __str__(self):
        return 'CHAOS'

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_path = self.items[index]['image_path']
        mask_path = self.items[index]['mask_path']

        image = Image.fromarray(pydicom.read_file(image_path).pixel_array).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        # 对tensor类型的mask进行处理
        mask = process_multiple_mask_tensor(mask, COLOR_MAP)

        return image, mask
