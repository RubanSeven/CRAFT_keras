# -*- coding: utf-8 -*-
# @Author: Ruban
# @License: Apache Licence
# @File: fake_util.py

import cv2
import math
import numpy as np


def watershed(src):
    """
    Performs a marker-based image segmentation using the watershed algorithm.
    :param src: 8-bit 1-channel image.
    :return: 32-bit single-channel image (map) of markers.
    """
    # cv2.imwrite('{}.png'.format(np.random.randint(1000)), src)
    gray = src.copy()
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # h, w = gray.shape[:2]
    # block_size = (min(h, w) // 4 + 1) * 2 + 1
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 0)
    _ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # dist_transform = opening & gray
    # cv2.imshow('dist_transform', dist_transform)
    # _ret, sure_bg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, cv2.THRESH_BINARY_INV)
    _ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, cv2.THRESH_BINARY)

    # Finding unknown region
    # sure_bg = np.uint8(sure_bg)
    sure_fg = np.uint8(sure_fg)
    # cv2.imshow('sure_fg', sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker label
    lingret, marker_map = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    marker_map = marker_map + 1

    # Now, mark the region of unknown with zero
    marker_map[unknown == 255] = 0

    marker_map = cv2.watershed(img, marker_map)

    return marker_map


def find_box(marker_map):
    """
    Calculate the minimum enclosing rectangles.
    :param marker_map: Input 32-bit single-channel image (map) of markers.
    :return: A list of point.
    """
    boxes = list()
    marker_count = np.max(marker_map)
    for marker_number in range(2, marker_count + 1):
        marker_cnt = np.swapaxes(np.array(np.where(marker_map == marker_number)), axis1=0, axis2=1)[:, ::-1]
        rect = cv2.minAreaRect(marker_cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        boxes.append(box)
    return boxes


def cal_distance(p1, p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))


def crop_image(src, points, dst_height=None):
    """
    Crop heat map with points.
    :param src: 8-bit single-channel image (map).
    :param points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    :return: dst_heat_map: Cropped image. 8-bit single-channel image (map) of heat map.
             src_points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
             dst_points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    """
    src_image = src.copy()
    src_points = np.float32(points)
    width = round((cal_distance(points[0], points[1]) + cal_distance(points[2], points[3])) / 2)
    height = round((cal_distance(points[1], points[2]) + cal_distance(points[3], points[0])) / 2)
    if dst_height is not None:
        ratio = dst_height / min(width, height)
        width = int(width * ratio)
        height = int(height * ratio)
    crop_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    perspective_mat = cv2.getPerspectiveTransform(src=src_points, dst=crop_points)
    dst_heat_map = cv2.warpPerspective(src_image, perspective_mat, (width, height),
                                       borderValue=0, borderMode=cv2.BORDER_CONSTANT)
    return dst_heat_map, src_points, crop_points


def un_warping(box, src_points, crop_points):
    """
    Unwarp the character bounding boxes.
    :param box: The character bounding box.
    :param src_points: Points before crop.
    :param crop_points: Points after crop.
    :return: The character bounding boxes after unwarp.
    """
    perspective_mat = cv2.getPerspectiveTransform(src=crop_points, dst=src_points)
    new_box = list()
    for x, y in box:
        new_x = int((perspective_mat[0][0] * x + perspective_mat[0][1] * y + perspective_mat[0][2]) /
                    (perspective_mat[2][0] * x + perspective_mat[2][1] * y + perspective_mat[2][2]))
        new_y = int((perspective_mat[1][0] * x + perspective_mat[1][1] * y + perspective_mat[1][2]) /
                    (perspective_mat[2][0] * x + perspective_mat[2][1] * y + perspective_mat[2][2]))
        new_box.append([new_x, new_y])
    return new_box


def enlarge_char_box(char_box, ratio):
    x_center, y_center = np.average(char_box[:, 0]), np.average(char_box[:, 1])
    char_box = char_box - [x_center, y_center]
    char_box = char_box * ratio
    char_box = char_box + [x_center, y_center]
    return char_box


def enlarge_char_boxes(char_boxes, crop_box):
    char_boxes = np.reshape(np.array(char_boxes), newshape=(-1, 4, 2))
    left, right, top, bottom = np.min(char_boxes[:, :, 0]), np.max(char_boxes[:, :, 0]), \
                               np.min(char_boxes[:, :, 1]), np.max(char_boxes[:, :, 1])
    width, height = crop_box[2, 0], crop_box[2, 1]
    offset = np.min([left, top, width - right, height - bottom])
    ratio = 1 + offset * 2 / min(width, height)
    char_boxes = np.array([enlarge_char_box(char_box, ratio) for char_box in char_boxes])
    char_boxes[:, :, 0] = np.clip(char_boxes[:, :, 0], 0, width)
    char_boxes[:, :, 1] = np.clip(char_boxes[:, :, 1], 0, height)
    return char_boxes


def divide_region(box, length):
    if length == 1:
        return [box]
    char_boxes = list()
    p1, p2, p3, p4 = box
    if cal_distance(p1, p2) + cal_distance(p3, p4) > cal_distance(p2, p3) + cal_distance(p4, p1):
        x_start1 = p1[0]
        y_start1 = p1[1]
        x_start2 = p4[0]
        y_start2 = p4[1]
        x_offset1 = (p2[0] - p1[0]) / length
        y_offset1 = (p2[1] - p1[1]) / length
        x_offset2 = (p3[0] - p4[0]) / length
        y_offset2 = (p3[1] - p4[1]) / length
    else:
        x_offset1 = (p4[0] - p1[0]) / length
        y_offset1 = (p4[1] - p1[1]) / length
        x_offset2 = (p3[0] - p2[0]) / length
        y_offset2 = (p3[1] - p2[1]) / length
        x_start1 = p1[0]
        y_start1 = p1[1]
        x_start2 = p2[0]
        y_start2 = p2[1]
    for i in range(length):
        char_boxes.append([
            [round(x_start1 + x_offset1 * i), round(y_start1 + y_offset1 * i)],
            [round(x_start1 + x_offset1 * (i + 1)), round(y_start1 + y_offset1 * (i + 1))],
            [round(x_start2 + x_offset2 * i), round(y_start2 + y_offset2 * i)],
            [round(x_start2 + x_offset2 * (i + 1)), round(y_start2 + y_offset2 * (i + 1))]
        ])

    return char_boxes


def cal_confidence(boxes, word_length):
    """
    Calculate the confidence score for the pseudo-GTs.
                (l(w) − min(l(w),|l(w) − lc(w)|))/l(w)
    l(w) is the word length of the sample w.
    lc(w) is the count of estimated character bounding boxes.
    :param boxes: The estimated character bounding boxes.
    :param word_length: The length of manually marked word.
    :return: Float. The confidence score for the  pseudo-GTs.
    """
    box_count = len(boxes)
    confidence = (word_length - min(word_length, abs(word_length - box_count))) / word_length
    return confidence


if __name__ == '__main__':
    region_score = cv2.imread(r'D:\python\CRAFT_keras\170.png', 0)
    markers = watershed(region_score)
    region_boxes = find_box(markers)
    result_img = cv2.cvtColor(region_score, cv2.COLOR_GRAY2BGR)
    for region_box in region_boxes:
        cv2.polylines(result_img, [region_box], True, color=(0, 0, 255))

    cv2.imshow('result', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
