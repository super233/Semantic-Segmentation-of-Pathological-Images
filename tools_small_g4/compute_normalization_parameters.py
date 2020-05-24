#!/usr/bin/python3
# -*- coding: utf-8 -*
# 计算用于数据标准化的mean，std参数
import os
import numpy as np
from PIL import Image
from torchvision import transforms

IMAGE_DIR_PATH = '/home/tangwenqi/workspace/data/small_g4_all/JPEGImages'

image_list = []

# transf = transforms.Resize((256, 256))

for name in os.listdir(IMAGE_DIR_PATH):

    if name.__contains__('.json'):
        continue

    image_path = os.path.join(IMAGE_DIR_PATH, name)

    # image_array = np.asarray(transf(Image.open(image_path).convert('RGB')), dtype=np.uint8)
    image_array = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.uint8)

    image_list.append(image_array)

    print('{} is ok'.format(image_path))

results = np.concatenate(image_list)
results = results.reshape((-1, 3))
results = results / 255

print('mean: {}'.format(np.mean(results, axis=0)))
print('std: {}'.format(np.std(results, axis=0)))
