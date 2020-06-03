import os
import os.path as osp
from tqdm import tqdm

from utils import makedirs

from PIL import Image
from torchvision import transforms

def resize_split(split_root="./data/split_40", target_root="./data/split_40_resize"):
    train_root = osp.join(split_root, "train")
    test_root = osp.join(split_root, "test")
    target_train_root = osp.join(target_root, "train")
    target_test_root = osp.join(target_root, "test")
    count = 0
    # resize train set
    for animal in os.listdir(train_root):
        makedirs(osp.join(target_train_root, animal))
        for img_name in tqdm(os.listdir(osp.join(train_root, animal)), desc=animal + " (train)", mininterval=1):
            img = Image.open(osp.join(train_root, animal, img_name))
            img = transforms.functional.resize(img, [224, 224])
            img.save(osp.join(target_train_root, animal, img_name))
            count += 1

    # resize test set
    for animal in os.listdir(test_root):
        makedirs(osp.join(target_test_root, animal))
        for img_name in tqdm(os.listdir(osp.join(test_root, animal)), desc=animal + " (test)", mininterval=1):
            img = Image.open(osp.join(test_root, animal, img_name))
            img = transforms.functional.resize(img, [224, 224])
            img.save(osp.join(target_test_root, animal, img_name))
            count += 1

    print("Process finished, {} images in total.".format(count))


if __name__ == '__main__':
    resize_split("./data/split_40", "./data/split_40_resize")

