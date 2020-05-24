#!/usr/bin/python3
# -*- coding: utf-8 -*
import pandas as pd
import os

BASE_PATH = '/home/tangwenqi/workspace/data/Skin-Lesion-Segmentation-ISIC-2017'
CSV_PATH = os.path.join(BASE_PATH, 'ISIC-2017_Training_Part3_GroundTruth.csv')

df = pd.read_csv(CSV_PATH)

for name in df['image_id']:
    image_path = os.path.join(BASE_PATH, 'ISIC-2017_Training_Data', '{}.jpg'.format(name))

    if not os.path.exists(image_path):
        print('{} not exists'.format(image_path))
print('ok')
