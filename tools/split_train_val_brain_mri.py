#!/usr/bin/python3
# -*- coding: utf-8 -*
# 划分Brain Mri数据集
import os
import json
import random

TRAIN_RATE = 0.7

BASE_PATH = '/home/tangwenqi/workspace/data/Brain-MRI-segmentation'
IMAGE_DIR_PATH = os.path.join(BASE_PATH, 'kaggle_3m')


def save_json(dir_list, json_path):
    json_data = []

    for dir in dir_list:
        for name in os.listdir(os.path.join(IMAGE_DIR_PATH, dir)):
            if name.__contains__('mask'):
                continue

            image_path = os.path.join(IMAGE_DIR_PATH, dir, name)
            mask_path = os.path.join(IMAGE_DIR_PATH, dir, '{}_mask.tif'.format(name.split('.tif')[0]))

            # print('image_path: {}'.format(image_path))
            # print('mask_path: {}'.format(mask_path))

            if not os.path.exists(mask_path):
                continue

            json_data.append({
                'image_path': image_path,
                'mask_path': mask_path
            })

    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    print('Save {}'.format(json_path))


train_json_path = os.path.join(BASE_PATH, 'train.json')
val_json_path = os.path.join(BASE_PATH, 'val.json')

dir_list = os.listdir(IMAGE_DIR_PATH)
random.shuffle(dir_list)

train_num = int(len(dir_list) * TRAIN_RATE)

train_dir_list = dir_list[:train_num]
val_dir_list = dir_list[train_num:]

save_json(train_dir_list, train_json_path)
save_json(val_dir_list, val_json_path)
