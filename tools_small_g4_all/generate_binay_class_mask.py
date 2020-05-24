#!/usr/bin/python3
# -*- coding: utf-8 -*
# 从多类的Mask图像内生成只有2个类别的Mask图像
from PIL import Image
import numpy as np
import shutil
import os

BASE_PATH = '/home/tangwenqi/workspace/data/small_g4_all'
SOURCE_DIR_PATH = os.path.join(BASE_PATH, 'Labels')
TARGET_DIR_PATH = os.path.join(BASE_PATH, 'Labels-binary')

# 创建目标目录，若存在，则删除后再创建
if os.path.exists(TARGET_DIR_PATH):
    shutil.rmtree(TARGET_DIR_PATH)
    print('Remove {}'.format(TARGET_DIR_PATH))
os.mkdir(TARGET_DIR_PATH)
print('Create {}'.format(TARGET_DIR_PATH))

for name in os.listdir(SOURCE_DIR_PATH):
    source_path = os.path.join(SOURCE_DIR_PATH, name)
    target_path = os.path.join(TARGET_DIR_PATH, name)

    image_array = np.array(Image.open(source_path))
    image_array[image_array != 0] = 255

    Image.fromarray(image_array).save(target_path)
    print('Save {}'.format(target_path))
