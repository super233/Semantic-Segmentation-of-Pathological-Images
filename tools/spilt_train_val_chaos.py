#!/usr/bin/python3
# -*- coding: utf-8 -*
import os
import json
import random

TRAIN_RATE = 0.7

BASE_PATH = '/home/tangwenqi/workspace/data/CHAOS'
DATA_PATH = os.path.join(BASE_PATH, 'Train_Sets', 'MR')

train_json_path = os.path.join(BASE_PATH, 'train.json')
val_json_path = os.path.join(BASE_PATH, 'val.json')


def save_json(dir_list, json_path):
    json_data = []

    for dir in dir_list:
        for name in os.listdir(os.path.join(DATA_PATH, dir, 'T2SPIR', 'DICOM_anon')):

            image_path = os.path.join(DATA_PATH, dir, 'T2SPIR', 'DICOM_anon', name)
            mask_path = os.path.join(DATA_PATH, dir, 'T2SPIR', 'Ground', '{}.png'.format(name.split('.dcm')[0]))

            if not os.path.exists(mask_path):
                continue

            json_data.append({
                'image_path': image_path,
                'mask_path': mask_path
            })

    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    print('Save {}'.format(json_path))


dir_list = os.listdir(DATA_PATH)
random.shuffle(dir_list)

train_num = int(len(dir_list) * TRAIN_RATE)

train_dir_list = dir_list[:train_num]
val_dir_list = dir_list[train_num:]

save_json(train_dir_list, train_json_path)
save_json(val_dir_list, val_json_path)
