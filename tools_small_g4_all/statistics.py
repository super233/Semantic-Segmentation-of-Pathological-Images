#!/usr/bin/python3
# -*- coding: utf-8 -*
# 从coco json中，统计每张图像内的类的情况
from pycocotools.coco import COCO
import json
import os
import numpy as np

NUM_CLASSES = 5

BASE_PATHS = ['/home/tangwenqi/workspace/data/small_g4/coco_train', '/home/tangwenqi/workspace/data/small_g4/coco_test']
TARGET_PATH = '/home/tangwenqi/workspace/data/small_g4_all_temp'

json_data = []
JSON_PATH = os.path.join(TARGET_PATH, 'statistics.json')


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_domain_type(image_path):
    """获取域的类型"""

    if image_path.__contains__('PASM'):
        return 'PASM'
    elif image_path.__contains__('Masson'):
        return 'Masson'
    elif image_path.__contains__('PAS'):
        return 'PAS'
    else:
        print('Error')
        exit(0)


def count_cls_num(anns):
    """统计每个类别的数量"""

    cls_num = np.zeros((NUM_CLASSES), dtype=np.int)
    for ann in anns:
        cls_num[ann['category_id']] += len(ann['segmentation'])

    return cls_num


for base_path in BASE_PATHS:
    coco = COCO(os.path.join(base_path, 'annotations.json'))
    # 获取image_id到Annotations的映射
    imgToAnns = coco.imgToAnns

    for image_id in coco.getImgIds():
        image_path = os.path.join(TARGET_PATH, coco.loadImgs(image_id)[0]['file_name'])
        # 如果对应图像不在目标目录内，则跳过剩下的操作
        if not os.path.exists(image_path):
            continue

        anns = imgToAnns[image_id]

        cls_num = count_cls_num(anns)

        json_data.append({
            'image_path': image_path,
            'main_class': np.argmax(cls_num),
            'class_num': cls_num,
            'domain_type': get_domain_type(image_path)
        })

        print('{}: {}'.format(image_path, cls_num))

with open(JSON_PATH, 'w') as f:
    json.dump(json_data, f, cls=NpEncoder)
print('Save {}'.format(JSON_PATH))
