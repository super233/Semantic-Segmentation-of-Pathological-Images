#!/usr/bin/python3
# -*- coding: utf-8 -*
# 划分Skin Lesion数据集
import os
import json
import random

TRAIN_RATE = 0.7

BASE_PATH = '/home/tangwenqi/workspace/data/Skin-Lesion-Segmentation-ISIC-2017'
IMAGE_DIR_PATH = os.path.join(BASE_PATH, 'ISIC-2017_Training_Data')
MASK_DIR_PATH = os.path.join(BASE_PATH, 'ISIC-2017_Training_Part1_GroundTruth')

train_json_path = os.path.join(BASE_PATH, 'train.json')
val_json_path = os.path.join(BASE_PATH, 'val.json')


def save_json(json_data, json_path):
    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    print('Save {}'.format(json_path))


json_data = []
for name in os.listdir(MASK_DIR_PATH):
    mask_path = os.path.join(MASK_DIR_PATH, name)
    image_path = os.path.join(IMAGE_DIR_PATH, '{}.jpg'.format(name.split('_segmentation.png')[0]))

    if not os.path.exists(image_path):
        continue

    json_data.append({
        'image_path': image_path,
        'mask_path': mask_path
    })

random.shuffle(json_data)

train_num = int(len(json_data) * TRAIN_RATE)
train_json = json_data[:train_num]
val_json = json_data[train_num:]

save_json(train_json, train_json_path)
save_json(val_json, val_json_path)
