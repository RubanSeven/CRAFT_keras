# -*- coding: utf-8 -*-
# @Author: Ruban
# @License: Apache Licence
# @File: box_util.py

import math
import numpy as np


def cal_slope(p1, p2):
    return (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-5)


def above_line(p, start_point, slope):
    y = (p[0] - start_point[0]) * slope + start_point[1]
    return p[1] < y


def reorder_points(point_list):
    """
    Reorder points of quadrangle.
    (top-left, top-right, bottom right, bottom left).
    :param point_list: List of point. Point is (x, y).
    :return: Reorder points.
    """
    # Find the first point which x is minimum.
    ordered_point_list = sorted(point_list, key=lambda x: (x[0], x[1]))
    first_point = ordered_point_list[0]

    # Find the third point. The slope is middle.
    slope_list = [[cal_slope(first_point, p), p] for p in ordered_point_list[1:]]
    ordered_slope_point_list = sorted(slope_list, key=lambda x: x[0])
    first_third_slope, third_point = ordered_slope_point_list[1]

    # Find the second point which is above the line between the first point and the third point.
    # All that's left is the fourth point.
    if above_line(ordered_slope_point_list[0][1], third_point, first_third_slope):
        second_point = ordered_slope_point_list[0][1]
        fourth_point = ordered_slope_point_list[2][1]
        reverse_flag = False
    else:
        second_point = ordered_slope_point_list[2][1]
        fourth_point = ordered_slope_point_list[0][1]
        reverse_flag = True

    # Find the top left point.
    second_fourth_slope = cal_slope(second_point, fourth_point)
    if first_third_slope < second_fourth_slope:
        if reverse_flag:
            reorder_point_list = [fourth_point, first_point, second_point, third_point]
        else:
            reorder_point_list = [second_point, third_point, fourth_point, first_point]
    else:
        reorder_point_list = [first_point, second_point, third_point, fourth_point]

    return reorder_point_list


def cal_min_box_distance(box1, box2):
    box_distance = [math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2) for p1 in box1 for p2 in box2]
    return np.min(box_distance)


def reorder_box(box_list):
    """
    Reorder character boxes.
    :param box_list: List of box. Box is a list of point. Point is (x, y).
    :return: Reorder boxes.
    """
    # Calculate the minimum distance between any two boxes.
    box_count = len(box_list)
    distance_mat = np.zeros((box_count, box_count), dtype=np.float32)
    for i in range(box_count):
        box1 = box_list[i]
        for j in range(i + 1, box_count):
            box2 = box_list[j]
            distance = cal_min_box_distance(box1, box2)
            distance_mat[i][j] = distance
            distance_mat[j][i] = distance

    # Find the boxes on the both ends.
    end_box_index = np.argmax(distance_mat)
    nan = distance_mat[end_box_index // box_count, end_box_index % box_count] + 1
    for i in range(box_count):
        distance_mat[i, i] = nan
    last_box_index = start_box_index = end_box_index // box_count
    last_box = box_list[start_box_index]

    # reorder box.
    reordered_box_list = [last_box]
    for i in range(box_count - 1):
        distance_mat[:, last_box_index] = nan
        closest_box_index = np.argmin(distance_mat[last_box_index])
        reordered_box_list.append(box_list[closest_box_index])
        last_box_index = closest_box_index

    return reordered_box_list


def cal_triangle_area(p1, p2, p3):
    """
    Calculate the area of triangle.
    S = |(x2 - x1)(y3 - y1) - (x3 - x1)(y2 - y1)| / 2
    :param p1: (x, y)
    :param p2: (x, y)
    :param p3: (x, y)
    :return: The area of triangle.
    """
    [x1, y1], [x2, y2], [x3, y3] = p1, p2, p3
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2


def cal_quadrangle_area(points):
    """
    Calculate the area of quadrangle.
    :return: The area of quadrangle.
    """
    points = reorder_points(points)
    p1, p2, p3, p4 = points
    s1 = cal_triangle_area(p1, p2, p3)
    s2 = cal_triangle_area(p3, p4, p1)
    s3 = cal_triangle_area(p2, p3, p4)
    s4 = cal_triangle_area(p4, p1, p2)
    if s1 + s2 == s3 + s4:
        return s1 + s2
    else:
        return 0


def cal_intersection(points):
    """
    Calculate the intersection of diagonals.
    x=[(x3-x1)(x4-x2)(y2-y1)+x1(y3-y1)(x4-x2)-x2(y4-y2)(x3-x1)]/[(y3-y1)(x4-x2)-(y4-y2)(x3-x1)]
    y=(y3-y1)[(x4-x2)(y2-y1)+(x1-x2)(y4-y2)]/[(y3-y1)(x4-x2)-(y4-y2)(x3-x1)]+y1
    :param points: (x1, y1), (x2, y2), (x3, y3), (x4, y4).
    :return: (x, y).
    """
    [x1, y1], [x2, y2], [x3, y3], [x4, y4] = points
    x = ((x3 - x1) * (x4 - x2) * (y2 - y1) + x1 * (y3 - y1) * (x4 - x2) - x2 * (y4 - y2) * (x3 - x1)) \
        / ((y3 - y1) * (x4 - x2) - (y4 - y2) * (x3 - x1) + 1e-5)
    y = (y3 - y1) * ((x4 - x2) * (y2 - y1) + (x1 - x2) * (y4 - y2)) \
        / ((y3 - y1) * (x4 - x2) - (y4 - y2) * (x3 - x1) + 1e-5) + y1
    return [x, y]


def cal_center_point(points):
    points = np.array(points)
    return [round(np.average(points[:, 0])), round(np.average(points[:, 1]))]


def cal_point_pairs(points):
    intersection = cal_intersection(points)
    p1, p2, p3, p4 = points
    point_pairs = [[cal_center_point([p1, p2, intersection]), cal_center_point([p3, p4, intersection])],
                   [cal_center_point([p2, p3, intersection]), cal_center_point([p4, p1, intersection])]]
    return point_pairs


def cal_affinity_box(point_pairs1, point_pairs2):
    areas = [cal_quadrangle_area([p1, p2, p3, p4]) for p1, p2 in point_pairs1 for p3, p4 in point_pairs2]
    max_area_index = np.argmax(areas)
    affinity_box = [point_pairs1[max_area_index // 2][0],
                    point_pairs1[max_area_index // 2][1],
                    point_pairs2[max_area_index % 2][0],
                    point_pairs2[max_area_index % 2][1]]
    return np.int32(affinity_box)


def cal_affinity_boxes(region_box_list, reorder_point_flag=True, reorder_box_flag=True):
    if reorder_point_flag:
        region_box_list = [reorder_points(region_box) for region_box in region_box_list]
    if reorder_box_flag:
        region_box_list = reorder_box(region_box_list)
    point_pairs_list = [cal_point_pairs(region_box) for region_box in region_box_list]
    affinity_box_list = list()
    for i in range(len(point_pairs_list) - 1):
        affinity_box = cal_affinity_box(point_pairs_list[i], point_pairs_list[i + 1])
        reorder_affinity_box = reorder_points(affinity_box)
        affinity_box_list.append(reorder_affinity_box)
    return affinity_box_list


if __name__ == '__main__':
    # import cv2
    # img = np.zeros((512, 512, 3), dtype=np.uint8)
    # pts = [(251,  96), (284, 112), (267, 112), (253, 118)]
    # for i in range(4):
    #     print(pts[i])
    #     cv2.line(img, pts[i], pts[(i + 1) % 4], (255, 0, 0))
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(reorder_points([[251, 96], [284, 112], [267, 112], [253, 118]]))
    print(reorder_points([[0, 0], [4, 4], [4, 0], [0, 4]]))
    print(reorder_points([[56, 25], [85, 45], [25, 80], [15, 45]]))

    print(reorder_box([[[0, 0], [4, 4], [4, 0], [0, 4]],
                       [[12, 0], [16, 4], [16, 0], [12, 4]],
                       [[16, 0], [20, 4], [20, 0], [16, 4]],
                       [[4, 0], [8, 4], [8, 0], [4, 4]],
                       [[8, 0], [12, 4], [12, 0], [8, 4]]]))

    print(cal_affinity_boxes([[[0, 0], [4, 4], [4, 0], [0, 4]],
                              [[12, 0], [16, 4], [16, 0], [12, 4]],
                              [[16, 0], [20, 4], [20, 0], [16, 4]],
                              [[4, 0], [8, 4], [8, 0], [4, 4]],
                              [[8, 0], [12, 4], [12, 0], [8, 4]]]))
