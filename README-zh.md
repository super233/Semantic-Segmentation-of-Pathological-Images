# 病理图像分割 Pytorch

基于Pytorch实现的病理图像分割，可作为语义分割的入门代码

## 数据集

- [Brain MRI Segmentation](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation)
- [Chest Xray Masks and Labels](https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels)
- [Skin Lesion Segmantation](https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a)
- [CHAOS](https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/)
- ~~Glomerulus~~

## 模型

- UNet
- UNet++
- Attention UNet
- ~~R2U-Net~~
- ~~Attention R2U-Net~~
- ~~NFNPlus~~
- ~~XNet~~

## 环境

- Pytorch 1.4.0
- Torchvision 0.5.0
- Python 3.7
- 一些其他的包请自行安装

## 准备

- 对于每份数据集，需要在其对应的类内修改其文件夹的绝对路径
- 每份数据集，需要生成`train.json`和`val.json`

## 使用

运行`train.py`，传入指定的命令行参数即可