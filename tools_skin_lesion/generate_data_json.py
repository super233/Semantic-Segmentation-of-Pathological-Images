#!/usr/bin/python3
# -*- coding: utf-8 -*
import os
import json
import pandas as pd

BASE_PATH = '/home/tangwenqi/workspace/data/Skin-Lesion-Segmentation-ISIC-2017'

CSV_PATH = os.path.join(BASE_PATH, 'ISIC-2017_Training_Part3_GroundTruth.csv')
IMAGE_DIR_PATH = os.path.join(BASE_PATH, 'ISIC-2017_Training_Data')
MASK_DIR_PATH = os.path.join(BASE_PATH, 'ISIC-2017_Training_Part1_GroundTruth')
JSON_DATA = os.path.join(BASE_PATH, 'data.json')


def get_class_id(melanoma, seborrheic_keratosis):
    if melanoma == 1:
        return 2
    if seborrheic_keratosis == 1:
        return 3
    else:
        return 1


json_data = []
df = pd.read_csv(CSV_PATH)
for index in df.index:
    data = df.loc[index]
    name = data['image_id']
    class_id = get_class_id(data['melanoma'], data['seborrheic_keratosis'])

    image_path = os.path.join(IMAGE_DIR_PATH, '{}.jpg'.format(name))
    mask_path = os.path.join(MASK_DIR_PATH, '{}_segmentation.png'.format(name))

    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        print('{} not exists'.format(name))
        continue

    json_data.append({
        'image_path': image_path,
        'mask_path': mask_path,
        'class': class_id
    })

with open(JSON_DATA, 'w') as f:
    json.dump(json_data, f)
print('Save {}'.format(JSON_DATA))
