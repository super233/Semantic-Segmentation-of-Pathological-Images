#!/usr/bin/python3
# -*- coding: utf-8 -*
import os
import json
import random

TRAINING_RATE = 0.7

TRANSFORM_TYPES = ['VerticalFlip', 'HorizontalFlip', 'Transpose', 'RandomRotate90']

BASE_PATH = '/home/tangwenqi/workspace/data/small_g4_all'
IMAGE_DIR_PATH = os.path.join(BASE_PATH, 'JPEGImages')
MASK_DIR_PATH = os.path.join(BASE_PATH, 'Labels')

STATISTICS_JSON_PATH = os.path.join(BASE_PATH, 'statistics.json')

split_file_by_cls = {
    1: [],
    2: [],
    3: [],
    4: []
}


def get_statistics_data(path):
    print('Loading statistics.json...')
    stat_data = {}
    with open(path) as f:
        json_data = json.load(f)

    for t in json_data:
        name = os.path.split(t['image_path'])[-1].split('.jpg')[0]
        stat_data[name] = t

    return stat_data


def is_augmented(path):
    for type in TRANSFORM_TYPES:
        if path.__contains__(type):
            return True
    return False


stat_data = get_statistics_data(STATISTICS_JSON_PATH)

for t_name in os.listdir(IMAGE_DIR_PATH):
    if not is_augmented(t_name):
        name = t_name.split('.jpg')[0]
        split_file_by_cls[stat_data[name]['main_class']].append(name)

train_json_path = os.path.join(BASE_PATH, 'train.json')
train_json = []
val_json_path = os.path.join(BASE_PATH, 'val.json')
val_json = []

for cls_id, name_list in split_file_by_cls.items():
    random.shuffle(name_list)
    train_num = int(len(name_list) * TRAINING_RATE)
    train_name_list = name_list[:train_num]
    val_name_list = name_list[train_num:]

    for name in train_name_list:
        train_json.append({
            'image_path': os.path.join(IMAGE_DIR_PATH, '{}.jpg'.format(name)),
            'mask_path': os.path.join(MASK_DIR_PATH, '{}.png'.format(name)),
            'main_class': cls_id
        })

        for type in TRANSFORM_TYPES:
            train_json.append({
                'image_path': os.path.join(IMAGE_DIR_PATH, '{}-{}.jpg'.format(name, type)),
                'mask_path': os.path.join(MASK_DIR_PATH, '{}-{}.png'.format(name, type)),
                'main_class': cls_id
            })

    for name in val_name_list:
        val_json.append({
            'image_path': os.path.join(IMAGE_DIR_PATH, '{}.jpg'.format(name)),
            'mask_path': os.path.join(MASK_DIR_PATH, '{}.png'.format(name)),
            'main_class': cls_id
        })

    print('class {} is ok, num: {}'.format(cls_id, len(name_list)))

with open(train_json_path, 'w') as f:
    json.dump(train_json, f)
print('Save {}'.format(train_json_path))

with open(val_json_path, 'w') as f:
    json.dump(val_json, f)
print('Save {}'.format(val_json_path))
