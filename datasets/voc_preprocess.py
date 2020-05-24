#!/usr/bin/python3
# -*- coding: utf-8 -*
# 对VOC数据集进行预处理，先将所有数据pad为500*500的，然后进行resize，并将resize产生的噪声标签替换为0
import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
import shutil
from torchvision import transforms

PAD_MAX_SIZE = 500
RESIZE_SIZE = 256


def create_dir(dir_path):
    """创建目录，若存在，则删除后再创建"""

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('{} has existed, remove it.'.format(dir_path))
    os.mkdir(dir_path)
    print('create dir {}'.format(dir_path))


def pad(image, pad_max_length, type):
    """在原始图像的四周添加0来将图像pad为指定的size"""

    assert type in ['image', 'mask']

    array = np.asarray(image)

    h, w = array.shape[:2]
    # 计算四周应该pad的距离
    top_pad_h = (pad_max_length - h) // 2
    bottom_pad_h = pad_max_length - h - top_pad_h
    left_pad_w = (pad_max_length - w) // 2
    right_pad_w = pad_max_length - w - left_pad_w

    if type == 'image':
        pad_array = np.pad(array, ((top_pad_h, bottom_pad_h), (left_pad_w, right_pad_w), (0, 0)))
    else:
        pad_array = np.pad(array, ((top_pad_h, bottom_pad_h), (left_pad_w, right_pad_w)))

    return Image.fromarray(pad_array)


def resize(image, transform, type):
    """将图像resize，同时清除掉resize带来的噪声标签"""

    assert type in ['image', 'mask']

    if type == 'image':
        return transform(image)
    else:
        # 正确的标签
        old_labels = np.unique(np.asarray(image))
        # resize后的标签，含有噪声
        image_resized_array = np.array(transform(image))
        # 将255也认为是噪声
        new_labels = np.append(np.unique(image_resized_array), 255)
        # 将噪声标签数据修改为0
        for n_l in new_labels:
            if n_l not in old_labels:
                image_resized_array[image_resized_array == n_l] = 0

        return Image.fromarray(image_resized_array)


def transform(image, pad_max_length, resize_size, type):
    image_transformed = pad(image, pad_max_length, type)
    image_transformed = resize(image_transformed, transforms.Resize((resize_size, resize_size)), type)
    return image_transformed


def run(mode):
    assert mode in ['train', 'val']

    if mode == 'train':
        image_dir_path = '/home/tangwenqi/workspace/data/benchmark_RELEASE/dataset/img'
        mask_dir_path = '/home/tangwenqi/workspace/data/benchmark_RELEASE/dataset/cls'
        txt_path = '/home/tangwenqi/workspace/data/benchmark_RELEASE/dataset/train.txt'
    else:
        image_dir_path = '/home/tangwenqi/workspace/data/VOCdevkit/VOC2012/JPEGImages'
        mask_dir_path = '/home/tangwenqi/workspace/data/VOCdevkit/VOC2012/SegmentationClass'
        txt_path = '/home/tangwenqi/workspace/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'

    image_transformed_dir_path = '{}-transformed'.format(image_dir_path)
    mask_transformed_dir_path = '{}-transformed'.format(mask_dir_path)
    create_dir(image_transformed_dir_path)
    create_dir(mask_transformed_dir_path)

    with open(txt_path) as f:
        file_names = [t.strip('\n') for t in f.readlines()]

    for name in file_names:
        image_path = os.path.join(image_dir_path, '{}.jpg'.format(name))
        image = Image.open(image_path).convert('RGB')

        if mode == 'train':
            mask_path = os.path.join(mask_dir_path, '{}.mat'.format(name))
            mask = Image.fromarray(loadmat(mask_path)['GTcls']['Segmentation'][0][0])
        else:
            mask_path = os.path.join(mask_dir_path, '{}.png'.format(name))
            mask = Image.open(mask_path)

        image_transformed = transform(image, PAD_MAX_SIZE, RESIZE_SIZE, 'image')
        image_transformed_path = os.path.join(image_transformed_dir_path, '{}.jpg'.format(name))
        image_transformed.save(image_transformed_path)
        print('save {}'.format(image_transformed_path))

        mask_transformed = transform(mask, PAD_MAX_SIZE, RESIZE_SIZE, 'mask')
        mask_transformed_path = os.path.join(mask_transformed_dir_path, '{}.png'.format(name))
        mask_transformed.save(mask_transformed_path)
        print('save {}'.format(mask_transformed_path))


if __name__ == '__main__':
    run('train')
    run('val')
