import cv2
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
import argparse
from utils import get_awa2_split_paths, makedirs


def detect_keypoint(detector, image_path, vis_save_path, des_save_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(gray, None)
    # visualize the annotations
    image_annot = cv2.drawKeypoints(image, kp, image, color=(0, 0, 255))
    cv2.imwrite(vis_save_path, image_annot)
    # save the descriptors
    np.save(des_save_path, des)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("OpenCV Detectors -- SIFT")
    parser.add_argument("--data-root", type=str, default='./data/split_40')
    parser.add_argument("--save-folder", type=str, default="./save-sift")
    parser.add_argument("--constraint", type=int, default=0, help="how many key points to detect")
    args = parser.parse_args()

    assert args.constraint >= 0
    if args.constraint == 0:
        detector = cv2.xfeatures2d.SIFT_create()
    else:
        detector = cv2.xfeatures2d.SIFT_create(args.constraint)
    all_class, all_image = get_awa2_split_paths(args.data_root, verbose=1)

    # For train set
    for label, image_paths in all_image['train'].items():
        vis_save_folder = osp.join(args.save_folder, f"./{args.constraint}/visualization/train/{label}")
        des_save_folder = osp.join(args.save_folder, f"./{args.constraint}/descriptor/train/{label}")
        makedirs(vis_save_folder)
        makedirs(des_save_folder)
        for image_path in tqdm(image_paths, desc=label + " (train)", mininterval=1):
            image_name = image_path.split("/")[-1]
            image_name = image_name.split(".")[0]
            vis_save_path = osp.join(vis_save_folder, f"{image_name}.png")
            des_save_path = osp.join(des_save_folder, f"{image_name}.npy")
            detect_keypoint(detector, image_path, vis_save_path, des_save_path)

    # For test set
    for label, image_paths in all_image['test'].items():
        vis_save_folder = osp.join(args.save_folder, f"./{args.constraint}/visualization/test/{label}")
        des_save_folder = osp.join(args.save_folder, f"./{args.constraint}/descriptor/test/{label}")
        makedirs(vis_save_folder)
        makedirs(des_save_folder)
        for image_path in tqdm(image_paths, desc=label + " (test)", mininterval=1):
            image_name = image_path.split("/")[-1]
            image_name = image_name.split(".")[0]
            vis_save_path = osp.join(vis_save_folder, f"{image_name}.png")
            des_save_path = osp.join(des_save_folder, f"{image_name}.npy")
            detect_keypoint(detector, image_path, vis_save_path, des_save_path)

    # for label, image_paths in all_image.items():
    #     vis_save_folder = osp.join(args.save_folder, f"./{args.constraint}/visualization/{label}")
    #     des_save_folder = osp.join(args.save_folder, f"./{args.constraint}/descriptor/{label}")
    #     makedirs(vis_save_folder)
    #     makedirs(des_save_folder)
    #     for image_path in tqdm(image_paths, desc=label, mininterval=1):
    #         image_name = image_path.split("/")[-1]
    #         image_name = image_name.split(".")[0]
    #         vis_save_path = osp.join(vis_save_folder, f"{image_name}.png")
    #         des_save_path = osp.join(des_save_folder, f"{image_name}.npy")
    #         detect_keypoint(detector, image_path, vis_save_path, des_save_path)