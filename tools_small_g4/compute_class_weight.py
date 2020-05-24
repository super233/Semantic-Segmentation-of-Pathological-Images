#!/usr/bin/python3
# -*- coding: utf-8 -*
# 计算每个类别的权值
from pycocotools.coco import COCO
import os
import numpy as np

NUM_CLASSES = 5
BASE_PATHS = ['/home/tangwenqi/workspace/data/small_g4/coco_train', '/home/tangwenqi/workspace/data/small_g4/coco_test']


def get_class_id_and_pixel_num(ann):
    class_id = ann['category_id']
    pixel_num = 0
    for t in ann['segmentation']:
        pixel_num += len(t) / 2

    return class_id, pixel_num


class_pixel_num = np.zeros((NUM_CLASSES))

for base_path in BASE_PATHS:
    coco_json_path = os.path.join(base_path, 'annotations.json')

    coco = COCO(coco_json_path)
    anns = coco.anns

    for ann in anns.values():
        class_id, frequency = get_class_id_and_pixel_num(ann)
        class_pixel_num[class_id] += frequency

    class_pixel_num[0] += len(coco.imgs) * 1024 * 1360

    print('{} is ok'.format(coco_json_path))

class_frequency = class_pixel_num / np.sum(class_pixel_num)
class_weight = np.median(class_frequency) / class_frequency

print('class_pixel_num: {}'.format(class_pixel_num))
print('class_frequency: {}'.format(class_frequency))
print('class_weight: {}'.format(class_weight))
