# -*- coding: utf-8 -*-
# @Author: Ruban
# @License: Apache Licence
# @File: img_util.py

import cv2
import numpy as np
from skimage import io
from utils.box_util import cal_affinity_boxes

# RGB
NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
NORMALIZE_VARIANCE = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


def load_image(img_path):
    """
    Load an image from file.
    :param img_path: Image file path, e.g. ``test.jpg`` or URL.
    :return: An RGB-image MxNx3.
    """
    img = io.imread(img_path)
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.array(img)

    return img


def img_normalize(src):
    """
    Normalize a RGB image.
    :param src: Image to normalize. Must be RGB order.
    :return: Normalized Image
    """
    img = src.copy().astype(np.float32)

    img -= NORMALIZE_MEAN
    img /= NORMALIZE_VARIANCE
    return img


def img_unnormalize(src):
    """
    Unnormalize a RGB image.
    :param src: Image to unnormalize. Must be RGB order.
    :return: Unnormalized Image.
    """
    img = src.copy()

    img *= NORMALIZE_VARIANCE
    img += NORMALIZE_MEAN

    return img.astype(np.uint8)


def img_resize(src, ratio, max_size, interpolation):
    """
    Resize image with a ratio.
    :param src: Image to resize.
    :param ratio: Scaling ratio.
    :param max_size: Maximum size of Image.
    :param interpolation: Interpolation method. See OpenCV document.
    :return: dst: Resized image.
             target_ratio: Actual scaling ratio.
    """
    img = src.copy()
    height, width, channel = img.shape

    target_ratio = min(max_size / max(height, width), ratio)
    target_h, target_w = int(height * target_ratio), int(width * target_ratio)
    dst = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    return dst, target_ratio


def score_to_heat_map(score):
    """
    Convert region score or affinity score to heat map.
    :param score: Region score or affinity score.
    :return: Heat map.
    """
    heat_map = (np.clip(score, 0, 1) * 255).astype(np.uint8)
    heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
    return heat_map


def create_affinity_box(boxes):
    affinity_boxes = cal_affinity_boxes(boxes)
    return affinity_boxes


def create_score_box(boxes_list):
    region_box_list = list()
    affinity_box_list = list()

    for boxes in boxes_list:
        region_box_list.extend(boxes)
        if len(boxes) > 0:
            affinity_box_list.extend(create_affinity_box(boxes))

    return region_box_list, affinity_box_list


def load_sample(img_path, img_size, word_boxes, boxes_list):
    img = load_image(img_path)

    height, width = img.shape[:2]
    ratio = img_size / max(height, width)
    target_height = int(height * ratio)
    target_width = int(width * ratio)
    img = cv2.resize(img, (target_width, target_height))

    normalized_img = img_normalize(img)
    # padding
    img = np.zeros((img_size, img_size, 3), dtype=np.float32)
    img[:target_height, :target_width] = normalized_img

    word_boxes = [[[int(x * ratio), int(y * ratio)] for x, y in box] for box in word_boxes]

    if len(boxes_list) == 0:
        return img, word_boxes, boxes_list, [], [], (target_width, target_height)

    boxes_list = [[[[int(x * ratio), int(y * ratio)] for x, y in box] for box in boxes] for boxes in boxes_list]
    region_box_list, affinity_box_list = create_score_box(boxes_list)

    return img, word_boxes, boxes_list, region_box_list, affinity_box_list, (target_width, target_height)


def to_heat_map(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img
