import os
import os.path as osp

import argparse
import pickle

def get_awa2_paths(awa2_root="./data/Animals_with_Attributes2", verbose=0):
    '''
    Given the root of AWA2, generate the image folder, all classes and all image paths
    :param awa2_root: the root of AWA2
    :param verbose: whether to print the information of the dataset or not
    :return: the image folder, all classes and all image paths (in a dictionary)
    '''
    image_folder = osp.join(awa2_root, "JPEGImages")
    all_class = os.listdir(image_folder)
    all_pics = {animal:os.listdir(osp.join(image_folder, animal)) for animal in all_class}
    for animal in all_class:
        for i in range(len(all_pics[animal])):
            all_pics[animal][i] = osp.join(image_folder, animal, all_pics[animal][i])
    if verbose != 0:
        print("AWA2 Dataset Information:")
        print(" - class number: {}".format(len(all_class)))
        print(" - total image number: {}".format(sum([len(x) for x in all_pics.values()])))
        for animal, img_list in all_pics.items():
            print("   - {}: {}".format(animal, len(img_list)))
    return image_folder, all_class, all_pics


def get_awa2_split_paths(root, verbose=0):
    all_class = os.listdir(osp.join(root, 'train'))
    all_pics = {
        'train': {animal: os.listdir(osp.join(root, 'train', animal)) for animal in all_class},
        "test": {animal: os.listdir(osp.join(root, 'test', animal)) for animal in all_class}
    }
    for animal in all_class:
        for i in range(len(all_pics['train'][animal])):
            all_pics['train'][animal][i] = osp.join(root, 'train', animal, all_pics['train'][animal][i])
    for animal in all_class:
        for i in range(len(all_pics['test'][animal])):
            all_pics['test'][animal][i] = osp.join(root, 'test', animal, all_pics['test'][animal][i])
    if verbose != 0:
        print("AWA2 Dataset Information:")
        print(" - class number: {}".format(len(all_class)))
        print(" - total train number: {}".format(sum([len(x) for x in all_pics['train'].values()])))
        print(" - total test number: {}".format(sum([len(x) for x in all_pics['test'].values()])))
    return all_class, all_pics



def makedirs(dirs):
    if not osp.exists(dirs):
        os.makedirs(dirs)


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Try - data directory information")
    parser.add_argument('--data-root', type=str, default="./data/Animals_with_Attributes2")
    args = parser.parse_args()

    image_folder, all_class, all_pics = get_awa2_paths(args.data_root, verbose=1)

    split_all_class, split_all_pics = get_awa2_split_paths('./data/split_40_resize', verbose=1)
    # print(all_pics)

