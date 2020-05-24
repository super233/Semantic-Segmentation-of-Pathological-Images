#!/usr/bin/python3
# -*- coding: utf-8 -*
import os
import json

BASE_PATH = '/home/tangwenqi/workspace/data/small_g4/'

TRAIN_IMAGE_PATH = os.path.join(BASE_PATH, 'coco_train', 'JPEGImages')
TRAIN_MASK_PATH = os.path.join(BASE_PATH, 'coco_train', 'Labels')

TEST_IMAGE_PATH = os.path.join(BASE_PATH, 'coco_test', 'JPEGImages')
TEST_MASK_PATH = os.path.join(BASE_PATH, 'coco_test', 'Labels')

items = []
for name in os.listdir(TRAIN_MASK_PATH):
    mask_path = os.path.join(TRAIN_MASK_PATH, name)
    image_path = os.path.join(TRAIN_IMAGE_PATH, '{}.jpg'.format(name.split('.png')[0]))

    items.append({
        'image_path': image_path,
        'mask_path': mask_path
    })

with open(os.path.join(BASE_PATH, 'train.json'), 'w') as f:
    json.dump(items, f)
print('Save {}'.format(os.path.join(BASE_PATH, 'train.json')))

items = []
for name in os.listdir(TEST_MASK_PATH):
    mask_path = os.path.join(TEST_MASK_PATH, name)
    image_path = os.path.join(TEST_IMAGE_PATH, '{}.jpg'.format(name.split('.png')[0]))

    items.append({
        'image_path': image_path,
        'mask_path': mask_path
    })

with open(os.path.join(BASE_PATH, 'val.json'), 'w') as f:
    json.dump(items, f)
print('Save {}'.format(os.path.join(BASE_PATH, 'val.json')))
