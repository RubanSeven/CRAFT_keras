# -*- coding: utf-8 -*-
# @Author: Ruban
# @License: Apache Licence
# @File: gaussian.py

import cv2
import numpy as np


def gaussian_2d():
    """
    Create a 2-dimensional isotropic Gaussian map.
    :return: a 2D Gaussian map. 1000x1000.
    """
    mean = 0
    radius = 2.5
    # a = 1 / (2 * np.pi * (radius ** 2))
    a = 1.
    x0, x1 = np.meshgrid(np.arange(-5, 5, 0.01), np.arange(-5, 5, 0.01))
    x = np.append([x0.reshape(-1)], [x1.reshape(-1)], axis=0).T

    m0 = (x[:, 0] - mean) ** 2
    m1 = (x[:, 1] - mean) ** 2
    gaussian_map = a * np.exp(-0.5 * (m0 + m1) / (radius ** 2))
    gaussian_map = gaussian_map.reshape(len(x0), len(x1))

    max_prob = np.max(gaussian_map)
    min_prob = np.min(gaussian_map)
    gaussian_map = (gaussian_map - min_prob) / (max_prob - min_prob)
    gaussian_map = np.clip(gaussian_map, 0., 1.)
    return gaussian_map


class GaussianGenerator:
    def __init__(self):
        self.gaussian_img = gaussian_2d()

    @staticmethod
    def perspective_transform(src, dst_shape, dst_points):
        """
        Perspective Transform
        :param src: Image to transform.
        :param dst_shape: Tuple of 2 intergers(rows and columns).
        :param dst_points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
        :return: Image after perspective transform.
        """
        img = src.copy()
        h, w = img.shape[:2]
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_points = np.float32(dst_points)
        perspective_mat = cv2.getPerspectiveTransform(src=src_points, dst=dst_points)
        dst = cv2.warpPerspective(img, perspective_mat, (dst_shape[1], dst_shape[0]),
                                  borderValue=0, borderMode=cv2.BORDER_CONSTANT)
        return dst

    def gen(self, score_shape, points_list):
        score_map = np.zeros(score_shape, dtype=np.float32)
        for points in points_list:
            tmp_score_map = self.perspective_transform(self.gaussian_img, score_shape, points)
            score_map = np.where(tmp_score_map > score_map, tmp_score_map, score_map)
        score_map = np.clip(score_map, 0, 1.)
        # score_map = score_map.astype(np.uint8)
        return score_map
