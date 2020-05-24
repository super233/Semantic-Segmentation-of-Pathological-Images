#!/usr/bin/python3
# -*- coding: utf-8 -*
import os
import json
import numpy as np
import argparse


# 对当前的Labelme JSON进行分析，对其中的数据字段进行检查或调整（修改、删除），并打印统计情况
# 当前默认假定 JSON的文件名 == 对应图片文件名
# 检查的内容有：
# 1.Labelme JSON中是否缺少必要的字段（imagePath、imageWidth、imageHeight、shapes、shapes-points、shapes-label）
# 2.imagePath：检查对应文件是否存在，检查JSON文件名和imagePath是否一样
# 3.shapes中的label：检查是否在给定的几个label里面
# 4.shapes中的points：检查是否有x >= width，y <= height的点；检查是否有生成bbox后，左上角点和右上角点是同一个点；检查是否有无法构成polygon的points
# 删除或者修改：
# 1.删除缺失字段的JSON文件
# 2.删除label和imagePath有问题的JSON文件
# 3.修改points有问题的JSON文件

class NpEncoder(json.JSONEncoder):
    """将numpy数据写入JSON时的序列化类"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class WrongLabelInfo():
    """错误的label的统计信息"""

    def __init__(self, label):
        self.label = label
        # label的出现次数
        self.num = 0
        # label所在的JSON文件名
        self.json_files = set()

    def add(self, file):
        self.num += 1
        self.json_files.add(file)

    def display(self):
        print('{}: {} times'.format(self.label, self.num))
        for f in self.json_files:
            print('\t{}'.format(f))


class AnalyzeLabelme():
    """对Labelme JSON进行整体分析，找出其中错误的JSON文件，并删除或者修改"""

    fields = {'imagePath': None, 'imageWidth': None, 'imageHeight': None, 'shapes': ('points', 'label')}

    def __init__(self, base_path, right_labels):
        """
        :param base_path: 存放图片和Lableme JSON的顶层文件夹
        :param right_labels: 存放正确的label的list
        """
        self.base_path = base_path
        self.right_labels = right_labels

        # 存储无效的label的字典，其中key为label，value为WrongLabelInfo类
        self.wrong_labels_dict = {}
        # 存储无效的imagePath的对应的JSON文件名
        self.wrong_imagePath_dict = {}
        # 存储points有问题的JSON文件名
        self.wrong_points_set = set()
        # 存储缺失必要的Labelme JSON字段的文件名和缺失的字段，其中key为文件名，value为缺失的字段的set
        self.lost_fields_dict = {}

        self.json_file_path = None
        self.dir_path = None
        self.labelme_json = None

    def run(self):
        print('<<<< Being analyzed >>>>')
        for dir in os.listdir(self.base_path):
            self.dir_path = os.path.join(self.base_path, dir)
            # 只对文件夹继续处理
            if os.path.isdir(self.dir_path):
                for file in os.listdir(self.dir_path):
                    if file.endswith('.json'):
                        self.json_file_path = os.path.join(self.base_path, dir, file)
                        with open(self.json_file_path) as f:
                            self.labelme_json = json.load(f)
                        self.check_fields()
                        self.check_label()
                        self.check_imagePath()
                        self.check_points()

        self.output()

    def check_fields(self):
        """检查当前的Labelme JSON中是否缺少必要的字段"""
        for main_f, sub_fs in self.fields.items():
            if main_f not in self.labelme_json:
                if self.json_file_path not in self.lost_fields_dict:
                    self.lost_fields_dict[self.json_file_path] = set()
                self.lost_fields_dict[self.json_file_path].add(main_f)
            else:
                if sub_fs is not None:
                    for fd in self.labelme_json[main_f]:
                        for sub_f in sub_fs:
                            if sub_f not in fd:
                                if self.json_file_path not in self.lost_fields_dict:
                                    self.lost_fields_dict[self.json_file_path] = set()
                                self.lost_fields_dict[self.json_file_path].add('{}-{}'.format(main_f, sub_f))

    def check_label(self):
        """检查当前Lableme JSON中的所有label是否正确，存储错误的label以及其出现的JSON文件名"""
        for s in self.labelme_json['shapes']:
            label = s['label']
            # 如果label不在给定的label内
            if label not in self.right_labels:
                # 检查dict是否有key为label的记录，没有就初始化
                if label not in self.wrong_labels_dict:
                    self.wrong_labels_dict[label] = WrongLabelInfo(label)
                self.wrong_labels_dict[label].add(self.json_file_path)

    def check_imagePath(self):
        """检查当前Labelme JSON的imagePath所对应的文件是否存在，imagePath与其JSON的文件名是否相同，存储错误的imagePath对应的JSON文件名"""
        # 使用imagePath构造对应图片文件的完整路径
        image_file_path = os.path.join(self.dir_path, self.labelme_json['imagePath'])
        # 如果对应文件不存在，或者如果imagePath与其JSON的文件名不同
        if not os.path.exists(image_file_path) or image_file_path.split('.')[0] != self.json_file_path.split('.')[0]:
            self.wrong_imagePath_dict[self.json_file_path] = self.labelme_json['imagePath']

    def check_points(self):
        """检查当前Labelme JSON的points中的点坐标是否符合规范，存储不规范的JSON文件名"""
        # 1.不满足0<= value < size的异常，要求点的x、y坐标小于width、height，这种情况可以纠正
        # 2.x_max <= x_min, y_max <= y_min的异常，要求bbox的左上角 全部小于 右下角坐标，出现的原因是构成区域的所有点在一条线上，或者只有1个点
        # 3.在转coco计算area时，需要在mask矩阵内绘制多边形，points内只有一个点无法绘制，1个或者2个点无法构成一个闭区域
        # 2和3的这种shape只有删除了
        for s in self.labelme_json['shapes']:
            if not self.is_points_right(s['points']) or not self.is_area_available(s['points']):
                self.wrong_points_set.add(self.json_file_path)

    def is_points_right(self, points):
        """检查points内的所有点的x、y坐标是否小于width、height"""
        points = np.array(points)
        try:
            wrong_x = np.argwhere(points[:, 0] >= self.labelme_json['imageWidth'])
            wrong_y = np.argwhere(points[:, 1] >= self.labelme_json['imageHeight'])

            if len(wrong_x) > 0 or len(wrong_y) > 0:
                return False
            else:
                return True
        except KeyError:
            return True

    def is_area_available(self, points):
        """检查points所表示的区域是否有效（点个数>2，所有点不在一条线上）"""
        points = np.array(points)
        x_max = np.max(points[:, 0])
        x_min = np.min(points[:, 0])
        y_max = np.max(points[:, 1])
        y_min = np.min(points[:, 1])

        if len(points) < 3 or x_max <= x_min or y_max <= y_min:
            return False
        else:
            return True

    def remove_files(self):
        """删除label和imagePath有问题的JSON"""
        for f in self.lost_fields_dict.keys():
            if os.path.exists(f):
                os.remove(f)
        for label_info in self.wrong_labels_dict.values():
            for f in label_info.json_files:
                if os.path.exists(f):
                    os.remove(f)
        for f in self.wrong_imagePath_dict.keys():
            # 当前的文件可能在之前被删除了
            if os.path.exists(f):
                os.remove(f)

    def correct_points(self):
        """调整points中不符合规范的点坐标"""
        # 删除无法构成标注区域的点（点个数<3，所有点在一条直线上）
        # 再调整x，y为height、width的点，将其在坐标上移动一个像素点，使其满足albumentations的要求
        for file_path in self.wrong_points_set:
            # 如果对应文件不存在（在之前被删除了），则跳过当前文件
            if not os.path.exists(file_path):
                continue

            with open(file_path) as f:
                current_labelme_json = json.load(f)
            new_shapes = []
            for s in current_labelme_json['shapes']:
                # 如果当前points的区域无效，则删除
                if not self.is_area_available(s['points']):
                    continue
                # 如果当前points的区域内有个别点有问题，则对其调整
                if not self.is_points_right(s['points']):
                    points = np.array(s['points'])
                    for row in np.argwhere(points[:, 0] >= current_labelme_json['imageWidth']):
                        points[row[0], 0] = current_labelme_json['imageWidth'] - 1
                    for row in np.argwhere(points[:, 1] >= current_labelme_json['imageHeight']):
                        points[row[0], 1] = current_labelme_json['imageHeight'] - 1
                    s['points'] = points
                # 存储shape
                new_shapes.append(s)
            # 替换成新的shapes
            current_labelme_json['shapes'] = new_shapes
            # 覆盖原文件内容
            json.dump(current_labelme_json, open(file_path, 'w'), indent=4, cls=NpEncoder)

    def output(self):
        """输出，并与用户交互，由用户绝对是否删除有问题的JSON文件"""
        # 打印缺失字段的文件信息
        if len(self.lost_fields_dict) == 0:
            print('<<<< All files are complete. >>>>')
        else:
            print('** {} files lost fields. **'.format(len(self.lost_fields_dict)))
            for f_name, fields in self.lost_fields_dict.items():
                print('\t{}\tlost fields: '.format(f_name), end='')
                for field in fields:
                    print(field, end=', ')
                print()
        # 打印label的数据
        if len(self.wrong_labels_dict) == 0:
            print('<<<< All labels are right. >>>>')
        else:
            print('** {} wrong labels. **'.format(len(self.wrong_labels_dict)))
            for label_info in self.wrong_labels_dict.values():
                label_info.display()
        # 打印imagePath的数据
        if len(self.wrong_imagePath_dict) == 0:
            print('<<<< All imagePaths are right. >>>>')
        else:
            print('** {} wrong imagePaths. **'.format(len(self.wrong_imagePath_dict)))
            for f_name, imagePath in self.wrong_imagePath_dict.items():
                print('\t{}\twrong imagePath: {}'.format(f_name, imagePath))
        # 打印points的数据
        if len(self.wrong_points_set) == 0:
            print('<<<< All points are right. >>>>')
        else:
            print('** {} wrong points. **'.format(len(self.wrong_points_set)))
            for f in self.wrong_points_set:
                print('\t{}'.format(f))
        if len(self.wrong_labels_dict) != 0 or len(self.wrong_imagePath_dict) != 0 or len(
                self.wrong_points_set) != 0 or len(self.lost_fields_dict) != 0:
            # 是否删除错误的JSON文件（具有错误的label、无效的imagePath），并删除和纠正错误的points
            while True:
                is_remove_and_correct = input('Delete wrong JSON files, remove and correct wrong points? y/n ').lower()
                if is_remove_and_correct == 'y' or is_remove_and_correct == 'n':
                    break
            if is_remove_and_correct == 'y':
                self.remove_files()
                self.correct_points()
                print('<<<< Success >>>>')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Analyze JSON files generated by Labelme to find the files with wrong "imagePath", 
        "label" and "points", then delete or correct them.''')
    # 添加必选参数“base_path”、“right_labels”，其中“right_labels”可以接收多个参数
    parser.add_argument('base_path', help='The top level director\'s path of labeled image files by Labelme.', type=str)
    parser.add_argument('right_labels', help='The right labels.', type=str, nargs='+')
    # 解析参数
    args = parser.parse_args()
    base_path = args.base_path
    # 保证right_labels中没有冗余的内容
    right_labels = set(args.right_labels)

    # 清除文件夹名字（路径）末尾多余的分隔符
    if base_path[-1] == '/' or base_path[-1] == '\\':
        base_path = base_path[:-1]
    analyzeLabelme = AnalyzeLabelme(base_path, right_labels)
    analyzeLabelme.run()
