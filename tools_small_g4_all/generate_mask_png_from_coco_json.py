#!/usr/bin/python3
# -*- coding: utf-8 -*
# 从coco json中读取数据，生成肾小球图像的标注数据
import os
from PIL import Image
import shutil
import numpy as np
from pycocotools.coco import COCO

# TODO: 记得修改路径
BASE_PATH = '/home/tangwenqi/workspace/data/small_g4/coco_train'
MASK_DIR_PATH = os.path.join(BASE_PATH, 'Labels')
COCO_JSON_PATH = os.path.join(BASE_PATH, 'annotations.json')

# 类别id与颜色值的映射
COLOR_MAP = {
    0: 0,
    1: 63,
    2: 127,
    3: 191,
    4: 255
}

# 创建存储Mask的文件夹，若存在，删除再创建
if os.path.exists(MASK_DIR_PATH):
    shutil.rmtree(MASK_DIR_PATH)
    print('Remove {}'.format(MASK_DIR_PATH))
os.mkdir(MASK_DIR_PATH)
print('Create {}'.format(MASK_DIR_PATH))

# 读取COCO JSON的内容
coco = COCO(COCO_JSON_PATH)
# 获取img_id与annotation的字典
imgToAnns = coco.imgToAnns

for img_id, img_json in coco.imgs.items():
    img_path = os.path.join(BASE_PATH, img_json['file_name'])
    # 如果图像不存在，则跳过
    if not os.path.exists(img_path):
        print('{} not exists.'.format(img_path))
        continue

    if len(imgToAnns[img_id]) == 0:
        print('########## {} has no annotation. ##########'.format(img_path))
        continue

    mask_path = os.path.join(MASK_DIR_PATH, '{}.png'.format(os.path.split(img_path)[-1].split('.jpg')[0]))

    mask_array = np.zeros((img_json['height'], img_json['width']), dtype=np.uint8)

    # 构造Mask数据
    for ann in imgToAnns[img_id]:
        mask_array += coco.annToMask(ann) * COLOR_MAP[ann['category_id']]

    # # 将对应的类别id替换为颜色值
    # for cls_id, value in COLOR_MAP.items():
    #     mask_array[mask_array == cls_id] = value

    Image.fromarray(mask_array).save(mask_path)
    print('Save {}'.format(mask_path))
