#!/usr/bin/python3
# -*- coding: utf-8 -*
# 计算Skin Lesion用于数据标准化的mean，std参数，因为尺寸不一，将其全部Resize为256*256后再计算
import os
import numpy as np
from PIL import Image

IMAGE_DIR_PATH = '/home/tangwenqi/workspace/data/Skin-Lesion-Segmentation-ISIC-2017/ISIC-2017_Training_Data'

image_list = []
for name in os.listdir(IMAGE_DIR_PATH):
    image_path = os.path.join(IMAGE_DIR_PATH, name)

    origin_image = Image.open(image_path).convert('RGB')
    image = origin_image.resize((256, 256), Image.BILINEAR)

    image_array = np.asarray(image, dtype=np.uint8)

    image_list.append(image_array)

    print('{} is ok'.format(image_path))

results = np.concatenate(image_list)
results = results.reshape((-1, 3))
results = results / 255

print('mean: {}'.format(np.mean(results, axis=0)))
print('std: {}'.format(np.std(results, axis=0)))
