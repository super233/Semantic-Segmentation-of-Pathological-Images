# Semantic Segmentation of Pathological Images

[中文](./README-zh.md)

Implemented by Pytorch, **it can also be a tutorial of semantic segmentation**.

If you like it, please star me. :smile:

## Datasets

There are some datasets from kaggle and other websites.

- [Brain MRI Segmentation](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)
- [Chest Xray Masks and Labels](https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels)
- [Skin Lesion Segmantation](https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a)
- [CHAOS](https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/)
- ~~Glomerulus~~

## Models

- UNet
- UNet++
- Attention UNet
- ~~R2U-Net~~
- ~~Attention R2U-Net~~
- ~~NFNPlus~~
- ~~XNet~~

## Requirements

- Pytorch 1.4.0
- Torchvision 0.5.0
- Python 3.7
- Some other libraries (find what you miss when running the code :-P)

## Preparations

- After downloading datasets, generate the `train.json` and `val.json` for each dataset with the following format:
```json
[
    {
        "image_path": "/home/tangwenqi/workspace/data/small_g4/coco_train/JPEGImages/20180550-PASM.jpg",
        "mask_path": "/home/tangwenqi/workspace/data/small_g4/coco_train/Labels/20180550-PASM.png"
    },
    {
        "image_path": "/home/tangwenqi/workspace/data/small_g4/coco_train/JPEGImages/20180620-PASM.jpg",
        "mask_path": "/home/tangwenqi/workspace/data/small_g4/coco_train/Labels/20180620-PASM.png"
    },
]
```
- For each dataset, modify its absolute path in corresponding class. 

## Usages

run `train.py` with some required and optional parameters, for more details please look `train.py`.
