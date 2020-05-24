#!/usr/bin/python3
# -*- coding: utf-8 -*
# 统计所有数据的形状
import os
from PIL import Image

BASE_PATH = '/home/tangwenqi/workspace/data/Chest-Xray-Masks-and-Labels/CXR_png'

shapes = {}

for name in os.listdir(os.path.join(BASE_PATH)):
    file_name = os.path.join(BASE_PATH, name)
    image = Image.open(file_name)

    s = image.size
    if s not in shapes:
        shapes[s] = 0
    shapes[s] += 1

for k, v in shapes.items():
    print('{}: {}'.format(k, v))
