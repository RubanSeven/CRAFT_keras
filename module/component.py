# -*- coding: utf-8 -*-
# @Author: Ruban
# @License: Apache Licence
# @File: component.py

import tensorflow as tf
import keras.backend as K


def _to_list(x):
    """
    Normalizes a list/tensor to a list.
    :param x: target object to be Normalized.
    :return: a list
    """
    if isinstance(x, list):
        return x
    return [x]


def _collect_input_shape(input_tensors):
    """
    Collect the output shape(s) of a list of keras tensors.
    :param input_tensors: list of input tensors (or single input tensor).
    :return: List of shape tuples (or single shape), one tuple per input.
    """
    input_tensors = _to_list(input_tensors)
    shapes = []
    for x in input_tensors:
        try:
            shapes.append(K.int_shape(x))
        except Exception as e:
            print(e)
            shapes.append(None)
    if len(shapes) == 1:
        return shapes[0]

    return shapes


def _permute_dimensions(x, pattern):
    """
    Permute axes in a tensor.
    :param x: Tensor or variable.
    :param pattern: A tuple of dimension indices, e.g. (0, 2, 1).
    :return: A Tensor
    """
    return tf.transpose(x, perm=pattern)
