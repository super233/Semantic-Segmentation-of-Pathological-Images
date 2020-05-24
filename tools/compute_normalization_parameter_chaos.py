#!/usr/bin/python3
# -*- coding: utf-8 -*
# 计算Chaos用于数据标准化的mean，std参数，因为尺寸不一，将其全部Resize为256*256后再计算
import os
import numpy as np
from PIL import Image
import pydicom

DATA_PATH = os.path.join('/home/tangwenqi/workspace/data/CHAOS/Train_Sets/MR')

image_list = []
for index in os.listdir(DATA_PATH):
    image_dir_path = os.path.join(DATA_PATH, index, 'T2SPIR', 'DICOM_anon')
    for name in os.listdir(image_dir_path):
        image_path = os.path.join(image_dir_path, name)

        origin_image = Image.fromarray(pydicom.read_file(image_path).pixel_array).convert('RGB')
        image = origin_image.resize((256, 256), Image.BILINEAR)

        image_array = np.asarray(image, dtype=np.uint8)

        image_list.append(image_array)

        print('{} is ok'.format(image_path))

results = np.concatenate(image_list)
results = results.reshape((-1, 3))
results = results / 255

print('mean: {}'.format(np.mean(results, axis=0)))
print('std: {}'.format(np.std(results, axis=0)))
