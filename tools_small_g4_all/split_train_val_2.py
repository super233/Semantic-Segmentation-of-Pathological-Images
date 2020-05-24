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


def is_augmented(path):
    for type in TRANSFORM_TYPES:
        if path.__contains__(type):
            return True
    return False


name_list = []
for name in os.listdir(IMAGE_DIR_PATH):
    if not is_augmented(name):
        name_list.append(name.split('.jpg')[0])

print('Total num: {}'.format(len(name_list)))

random.shuffle(name_list)

train_num = int(len(name_list) * TRAINING_RATE)

train_name_list = name_list[:train_num]
val_name_list = name_list[train_num:]

train_json_path = os.path.join(BASE_PATH, 'train.json')
train_json = []
for name in train_name_list:
    train_json.append({
        'image_path': os.path.join(IMAGE_DIR_PATH, '{}.jpg'.format(name)),
        'mask_path': os.path.join(MASK_DIR_PATH, '{}.png'.format(name))
    })

    for type in TRANSFORM_TYPES:
        train_json.append({
            'image_path': os.path.join(IMAGE_DIR_PATH, '{}-{}.jpg'.format(name, type)),
            'mask_path': os.path.join(MASK_DIR_PATH, '{}-{}.png'.format(name, type))
        })

val_json_path = os.path.join(BASE_PATH, 'val.json')
val_json = []
for name in val_name_list:
    val_json.append({
        'image_path': os.path.join(IMAGE_DIR_PATH, '{}.jpg'.format(name)),
        'mask_path': os.path.join(MASK_DIR_PATH, '{}.png'.format(name))
    })

with open(train_json_path, 'w') as f:
    json.dump(train_json, f)
print('Save {}'.format(train_json_path))

with open(val_json_path, 'w') as f:
    json.dump(val_json, f)
print('Save {}'.format(val_json_path))
