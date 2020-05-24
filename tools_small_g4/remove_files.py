#!/usr/bin/python3
# -*- coding: utf-8 -*
# 删除small_g1内的所有增广图像以及其mask
import os

TRANSFORM_TYPES = ['VerticalFlip', 'HorizontalFlip', 'Transpose', 'RandomRotate90']

BASE_PATHS = ['/home/tangwenqi/workspace/data/small_g4/coco_train', '/home/tangwenqi/workspace/data/small_g4/coco_test']


def is_augmented(path):
    for type in TRANSFORM_TYPES:
        if path.__contains__(type):
            return True
    return False


for base_path in BASE_PATHS:
    image_dir_path = os.path.join(base_path, 'JPEGImages')
    mask_dir_path = os.path.join(base_path, 'Labels')

    for name in os.listdir(image_dir_path):
        if not is_augmented(name):
            continue

        image_path = os.path.join(image_dir_path, name)
        mask_path = os.path.join(mask_dir_path, '{}.png'.format(name.split('.jpg')[0]))

        os.remove(image_path)
        print('Remove {}'.format(image_path))

        os.remove(mask_path)
        print('Remove {}'.format(mask_path))
