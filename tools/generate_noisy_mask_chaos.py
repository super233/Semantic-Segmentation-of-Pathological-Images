#!/usr/bin/python3
# -*- coding: utf-8 -*
import sys

sys.path.append('/home/tangwenqi/workspace/pycharm_workspace/Semantic-Segmentation-of-Pathological-Images')

import os
from PIL import Image
import numpy as np
import random
from utils.tools import add_noisy_triangle_label

BASE_PATH = '/home/tangwenqi/workspace/data/CHAOS'
DATA_PATH = os.path.join(BASE_PATH, 'Train_Sets', 'MR')
COLOR_MAP = {
    63: 1,
    126: 2,
    189: 3,
    252: 4
}

color_set = set(COLOR_MAP.keys())


def sy(input_color):
    '''生成对应的齐次噪声颜色'''

    # 集和运算求解剩下的所有颜色
    other_color = list(color_set - {input_color})

    p = np.random.uniform()
    if p >= 0.66:
        return other_color[2]
    elif p >= 0.33:
        return other_color[1]
    else:
        return other_color[0]


def asy(input_color):
    '''生成对应的非齐次噪声颜色'''

    output_color = input_color + 63

    if output_color > 252:
        return 63
    else:
        return output_color


def generate_noisy_mask(mask, type):
    mask_array = np.asarray(mask)
    origin_colors = []
    for color in COLOR_MAP.keys():
        if color in mask_array:
            origin_colors.append(color)

    # 如果没有标注则不生成噪声
    if len(origin_colors) == 0:
        return None
    else:
        noisy_colors = []
        if type == 'sy':
            for color in origin_colors:
                noisy_colors.append(sy(color))
        else:
            for color in origin_colors:
                noisy_colors.append(asy(color))

        noisy_mask_array = mask_array.copy()
        for i in range(len(noisy_colors)):
            noisy_mask_array[mask_array == origin_colors[i]] = noisy_colors[i]

        return Image.fromarray(noisy_mask_array)


# def generate_noisy_mask(mask):
#     mask_array = np.asarray(mask)
#     origin_colors = []
#     for color in COLOR_MAP.keys():
#         if color in mask_array:
#             origin_colors.append(color)
#
#     # # 如果没有标注，则随机产生三角形标注区域
#     # if len(origin_colors) == 0:
#     #     return add_noisy_triangle_label(mask)
#     # # 如果只有一个类别的标注，返回没有标注的mask
#     # elif len(origin_colors) == 1:
#     #     return Image.fromarray(np.zeros(mask.size, dtype=np.uint8))
#
#     # 如果没有标注或者只有1个类别的标注，则不生成噪声
#     if len(origin_colors) == 1 or len(origin_colors) == 0:
#         return None
#     else:
#         noisy_colors = origin_colors.copy()
#         # 如果只有2个类别，则将其交换
#         if len(noisy_colors) == 2:
#             noisy_colors.reverse()
#         else:
#             while noisy_colors == origin_colors:
#                 random.shuffle(noisy_colors)
#
#         noisy_mask_array = mask_array.copy()
#         for i in range(len(noisy_colors)):
#             noisy_mask_array[mask_array == origin_colors[i]] = noisy_colors[i]
#
#         return Image.fromarray(noisy_mask_array)


for index in os.listdir(DATA_PATH):
    for name in os.listdir(os.path.join(DATA_PATH, index, 'T2SPIR', 'Ground')):
        mask_path = os.path.join(DATA_PATH, index, 'T2SPIR', 'Ground', name)

        # 生成噪声
        for type in ['sy', 'asy']:
            noisy_mask_path = '{}_noisy_{}.png'.format(mask_path.split('.png')[0], type)

            mask = Image.open(mask_path).convert('L')
            noisy_mask = generate_noisy_mask(mask, type)

            if noisy_mask is not None:
                noisy_mask.save(noisy_mask_path)
                print('Save {}'.format(noisy_mask_path))

        # # 删除噪声图像的代码
        # if name.__contains__('noisy'):
        #     path = os.path.join(DATA_PATH, index, 'T2SPIR', 'Ground', name)
        #     os.remove(path)
        #     print('Remove {}'.format(path))
