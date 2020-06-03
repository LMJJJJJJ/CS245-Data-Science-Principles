import os
import os.path as osp
import shutil
import random
from utils import makedirs


def split_dataset(awa2_root="./data/Animals_with_Attributes2", test_set=40, verbose=0):
    image_cnt = 0
    target_image_folder = f"./data/split_{test_set}"
    train_image_folder = osp.join(target_image_folder, "train")
    test_image_folder = osp.join(target_image_folder, "test")
    test_set = test_set / 100

    original_image_folder = osp.join(awa2_root, "JPEGImages")
    all_class = os.listdir(original_image_folder)
    all_pics = {animal: os.listdir(osp.join(original_image_folder, animal)) for animal in all_class}

    for label, image_files in all_pics.items():
        makedirs(osp.join(train_image_folder, label))
        makedirs(osp.join(test_image_folder, label))
        random.shuffle(image_files)
        size = len(image_files)
        train_split = image_files[int(0.4 * size):]
        test_split = image_files[:int(0.4 * size)]
        for file in train_split:
            image_cnt += 1
            shutil.copy(osp.join(original_image_folder, label, file), osp.join(train_image_folder, label, file))
        for file in test_split:
            image_cnt += 1
            shutil.copy(osp.join(original_image_folder, label, file), osp.join(test_image_folder, label, file))
        if verbose != 0:
            print(f"Total images: {image_cnt}")


if __name__ == '__main__':
    split_dataset()

