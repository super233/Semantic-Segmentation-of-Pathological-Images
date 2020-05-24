#!/usr/bin/python3
# -*- coding: utf-8 -*
# 计算用于数据标准化的mean，std参数
import os
import numpy as np
from PIL import Image

IMAGE_DIR_PATH = '/home/tangwenqi/workspace/data/small_g4_all/JPEGImages'


def is_augmented(path):
    TRANSFORM_TYPES = ['VerticalFlip', 'HorizontalFlip', 'Transpose', 'RandomRotate90']
    for type in TRANSFORM_TYPES:
        if path.__contains__(type):
            return True
    return False


image_list = []
for name in os.listdir(IMAGE_DIR_PATH):
    image_path = os.path.join(IMAGE_DIR_PATH, name)

    if is_augmented(image_path):
        continue

    # image_array = np.asarray(transf(Image.open(image_path).convert('RGB')), dtype=np.uint8)

    # 将图像转为灰度图像
    image_array = np.asarray(Image.open(image_path).convert('L'), dtype=np.uint8)

    image_list.append(image_array)

    print('{} is ok'.format(image_path))

# results = np.concatenate(image_list)
# results = results.reshape((-1, 3))
# results = results / 255
#
# print('mean: {}'.format(np.mean(results, axis=0)))
# print('std: {}'.format(np.std(results, axis=0)))


results = np.concatenate(image_list)
results = results.flatten()
results = results / 255

print('mean: {}'.format(np.mean(results)))
print('std: {}'.format(np.std(results)))
