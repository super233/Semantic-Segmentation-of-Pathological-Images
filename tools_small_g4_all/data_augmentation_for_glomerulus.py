#!/usr/bin/python3
# -*- coding: utf-8 -*
# 对glomerulus数据集进行数据增广
import cv2
import os
import albumentations as A
import numpy as np

BASE_PATH = '/home/tangwenqi/workspace/data/small_g4_all'
TRANSFORM_TYPES = ['VerticalFlip', 'HorizontalFlip', 'Transpose', 'RandomRotate90']


def is_image_equal(image1, image2):
    """判断2张图像是否相等"""

    temp = np.asarray(image1 != image2)
    return temp.sum() == 0


def get_aug(image, type):
    height = image.shape[0]
    width = image.shape[1]

    if type == 'VerticalFlip':
        return A.VerticalFlip(p=1)
    elif type == 'HorizontalFlip':
        return A.HorizontalFlip(p=1)
    elif type == 'Transpose':
        return A.Compose([A.Transpose(p=1), A.Resize(height=height, width=width, p=1)])
    elif type == 'RandomRotate90':
        return A.Compose([A.RandomRotate90(p=1), A.Resize(height=height, width=width, p=1)])


def transform(image, mask, image_path, mask_path, type):
    aug = get_aug(image, type)
    augmented = aug(image=image, mask=mask)

    while type == 'RandomRotate90' and is_image_equal(mask, augmented['mask']):
        augmented = aug(image=image, mask=mask)

    save_image(augmented['image'], '{}-{}.jpg'.format(image_path.split('.jpg')[0], type))
    save_image(augmented['mask'], '{}-{}.png'.format(mask_path.split('.png')[0], type))


def save_image(image, save_path):

    # 如果是RGB数据，要将格式还原为BGR，不然保存后和原图的通道是相反的
    if save_path.__contains__('.jpg'):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, image)
    print('Save {}'.format(save_path))


def run(base_path):
    image_dir_path = os.path.join(base_path, 'JPEGImages')
    mask_dir_path = os.path.join(base_path, 'Labels')

    for name in os.listdir(mask_dir_path):
        mask_path = os.path.join(mask_dir_path, name)
        image_path = os.path.join(image_dir_path, '{}.jpg'.format(name.split('.png')[0]))

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        for type in TRANSFORM_TYPES:
            transform(image, mask, image_path, mask_path, type)


if __name__ == '__main__':
    run(BASE_PATH)
