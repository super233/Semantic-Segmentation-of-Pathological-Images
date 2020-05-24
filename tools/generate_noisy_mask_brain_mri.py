#!/usr/bin/python3
# -*- coding: utf-8 -*
import sys

sys.path.append('/home/tangwenqi/workspace/pycharm_workspace/Semantic-Segmentation-of-Pathological-Images')

import os
from PIL import Image
from utils.tools import add_noisy_triangle_label

BASE_PATH = '/home/tangwenqi/workspace/data/Brain-MRI-segmentation/kaggle_3m'

for dir in os.listdir(BASE_PATH):
    for name in os.listdir(os.path.join(BASE_PATH, dir)):
        if not name.__contains__('mask'):
            continue

        mask_path = os.path.join(BASE_PATH, dir, name)
        noisy_mask_path = '{}_noisy.tif'.format(mask_path.split('.tif')[0])

        mask = Image.open(mask_path).convert('L')
        noisy_mask = add_noisy_triangle_label(mask)

        noisy_mask.save(noisy_mask_path)
        print('Save {}'.format(noisy_mask_path))
