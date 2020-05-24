#!/usr/bin/python3
# -*- coding: utf-8 -*
# 删除small_g1_all内的所有增广图像以及其mask
import os

TRANSFORM_TYPES = ['VerticalFlip', 'HorizontalFlip', 'Transpose', 'RandomRotate90']

BASE_PATH = '/home/tangwenqi/workspace/data/small_g4_all_temp'
IMAGE_DIR_PATH = os.path.join(BASE_PATH, 'JPEGImages')
MASK_DIR_PATH = os.path.join(BASE_PATH, 'Labels')


def is_augmented(path):
    for type in TRANSFORM_TYPES:
        if path.__contains__(type):
            return True
    return False


for name in os.listdir(IMAGE_DIR_PATH):
    if not is_augmented(name):
        continue

    image_path = os.path.join(IMAGE_DIR_PATH, name)
    mask_path = os.path.join(MASK_DIR_PATH, '{}.png'.format(name.split('.jpg')[0]))

    os.remove(image_path)
    print('Remove {}'.format(image_path))

    os.remove(mask_path)
    print('Remove {}'.format(mask_path))
