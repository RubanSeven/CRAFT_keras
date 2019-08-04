# -*- coding: utf-8 -*-
# @Author: Ruban
# @License: Apache Licence
# @File: fake.py

import cv2
import traceback
import numpy as np
import tensorflow as tf
from keras.layers import Layer
from .component import _collect_input_shape
from utils.gaussian import GaussianGenerator
from utils.box_util import cal_affinity_boxes, reorder_points
from utils.fake_util import crop_image, watershed, find_box, un_warping, cal_confidence, divide_region, \
    enlarge_char_boxes


def create_pseudo_gt(word_boxes, word_lengths, region, affinity, confidence, pred_region):
    """
    Generate character boxes from each word-level annotation in a weakly-supervised manner.
    In order to reflect the reliability of the interim model’s prediction,
    the value of the confidence map over each word box is computed proportional to the number of
    the detected characters divided by the number of the ground truth characters,
    which is used for the learning weight during training.
    :param word_boxes: (word boxes, word_length).
    :param word_lengths: region map.
    :param region: region map.
    :param affinity: affinity map.
    :param confidence: confidence map.
    :param pred_region: region map.
    :return: region map, affinity map, confidence map.
    """
    gaussian_generator = GaussianGenerator()
    for word_box, word_length in zip(word_boxes, word_lengths):
        if word_length > 0:
            try:
                word_box = reorder_points(word_box)
                heat_map, src_points, crop_points = crop_image(pred_region * 255., word_box)
                heat_map = heat_map.astype(np.uint8)
                marker_map = watershed(heat_map)
                region_boxes = find_box(marker_map)

                confidence_value = cal_confidence(region_boxes, word_length)
                if confidence_value <= 0.5:
                    confidence_value = 0.5
                    region_boxes = divide_region(word_box, word_length)
                    region_boxes = [reorder_points(region_box) for region_box in region_boxes]
                else:
                    region_boxes = enlarge_char_boxes(region_boxes, crop_points)
                    region_boxes = [un_warping(region_box, src_points, crop_points) for region_box in region_boxes]

                tmp_confidence_mask = np.zeros(confidence.shape[:2], dtype=np.uint8)
                cv2.fillPoly(tmp_confidence_mask, [np.int32(word_box)], 1)
                tmp_confidence = tmp_confidence_mask.astype(np.float32) * confidence_value
                confidence = (1 - tmp_confidence_mask) * confidence + tmp_confidence

                tmp_region = np.float32(gaussian_generator.gen(region.shape[:2], region_boxes))
                region = np.where(tmp_region > 0, tmp_region, region)

                affinity_boxes = cal_affinity_boxes(region_boxes)
                tmp_affinity = np.float32(gaussian_generator.gen(affinity.shape[:2], affinity_boxes))
                affinity = np.where(tmp_affinity > 0, tmp_affinity, affinity)
            except Exception as e:
                print(e)
                traceback.print_exc()
        else:
            break

    return region, affinity, confidence


class Fake(Layer):
    def __init__(self, input_box, input_word_length, input_region, input_affinity, input_confidence, **kwargs):
        super(Fake, self).__init__(**kwargs)
        self.input_box = input_box
        self.input_word_length = input_word_length
        self.input_region = input_region
        self.input_affinity = input_affinity
        self.input_confidence = input_confidence

    def compute_output_shape(self, input_shape):
        output_shape = [_collect_input_shape(self.input_region),
                        _collect_input_shape(self.input_affinity),
                        _collect_input_shape(self.input_confidence),
                        ]
        return output_shape

    def call(self, inputs, **kwargs):
        # new_region, new_affinity, new_confidence = \
        #     tf.py_func(func=batch_create_pseudo_gt,
        #                inp=[self.input_box, self.input_word_length, self.input_region, self.input_affinity,
        #                     self.input_confidence, inputs],
        #                Tout=[tf.float32, tf.float32, tf.float32]
        #                )

        def tf_create_pseudo_gt(input_box, input_word_length, input_region, input_affinity,
                                input_confidence, region_pred):
            return tf.py_func(func=create_pseudo_gt,
                              inp=[input_box, input_word_length, input_region, input_affinity,
                                   input_confidence, region_pred],
                              Tout=[tf.float32, tf.float32, tf.float32]
                              )

        new_region, new_affinity, new_confidence = \
            tf.map_fn(lambda x: tf_create_pseudo_gt(x[0], x[1], x[2], x[3], x[4], x[5]),
                      elems=[self.input_box, self.input_word_length, self.input_region, self.input_affinity,
                             self.input_confidence, inputs],
                      dtype=[tf.float32, tf.float32, tf.float32])

        new_region.set_shape(_collect_input_shape(self.input_region))
        new_affinity.set_shape(_collect_input_shape(self.input_affinity))
        new_confidence.set_shape(_collect_input_shape(self.input_confidence))

        return [new_region, new_affinity, new_confidence]
