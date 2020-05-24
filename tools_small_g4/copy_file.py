import os
import shutil

TRANSFORM_TYPES = ['VerticalFlip', 'HorizontalFlip', 'Transpose', 'RandomRotate90']

ORIGINAL_PATHS = ['/home/tangwenqi/workspace/data/small_g4/coco_train',
                  '/home/tangwenqi/workspace/data/small_g4/coco_test']

TARGET_PATH = '/home/tangwenqi/workspace/data/small_g4_all'


def is_augmented(path):
    for type in TRANSFORM_TYPES:
        if path.__contains__(type):
            return True
    return False


for origin_path in ORIGINAL_PATHS:
    image_dir_path = os.path.join(origin_path, 'JPEGImages')
    mask_dir_path = os.path.join(origin_path, 'Labels')

    for name in os.listdir(mask_dir_path):

        if is_augmented(name):
            continue

        mask_path = os.path.join(mask_dir_path, name)
        image_path = os.path.join(image_dir_path, '{}.jpg'.format(name.split('.png')[0]))

        target_mask_path = os.path.join(TARGET_PATH, 'Labels', name)
        target_image_path = os.path.join(TARGET_PATH, 'JPEGImages', '{}.jpg'.format(name.split('.png')[0]))

        shutil.copy(mask_path, target_mask_path)
        print('Copy {} to {}'.format(mask_path, target_mask_path))

        shutil.copy(image_path, target_image_path)
        print('Copy {} to {}'.format(image_path, target_image_path))
