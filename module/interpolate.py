# -*- coding: utf-8 -*-
# @Author: Ruban
# @License: Apache Licence
# @File: interpolate.py

import tensorflow as tf
from keras.utils import conv_utils
from keras.engine import Layer, InputSpec
from .component import _collect_input_shape, _permute_dimensions


def _resie_image(x, target_layer, target_shape, data_format):
    """
    Resize the images contained in a 4D tensor.
    :param x: Tensor or variable to resize.
    :param target_layer: Tensor or variable. Resize the images to the same size as it is.
    :param target_shape: Tuple of 2 intergers(rows and columns).
    :param data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    :return: 4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, upsampled_rows, upsampled_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, upsampled_rows, upsampled_cols)`
    """
    if data_format == 'channels_first':
        new_shape = tf.shape(target_layer)[2:]
        x = _permute_dimensions(x, [0, 2, 3, 1])
        x = tf.image.resize_nearest_neighbor(x, new_shape)
        x = _permute_dimensions(x, [0, 3, 1, 2])
        x.set_shape((None, None, target_shape[2], target_shape[3]))
        return x
    elif data_format == 'channels_last':
        new_shape = tf.shape(target_layer)[1:3]
        x = tf.image.resize_nearest_neighbor(x, new_shape)
        x.set_shape((None, target_shape[1], target_shape[2], None))
        return x
    else:
        raise ValueError('Unknown data_format: ' + str(data_format))


class Interpolate(Layer):
    """
    UpSampling layer
    # Input shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, rows, cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, rows, cols)`

    # Output shape
        4D tensor with shape:
        - If `data_format` is `"channels_last"`:
            `(batch, upsampled_rows, upsampled_cols, channels)`
        - If `data_format` is `"channels_first"`:
            `(batch, channels, upsampled_rows, upsampled_cols)`
    """

    def __init__(self, target_layer, data_format=None, **kwargs):
        """
        :param target_layer: Tensor or variable. Resize the images to the same size as it is.
        :param data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        :param kwargs:
        """
        super(Interpolate, self).__init__(**kwargs)
        self.target_layer = target_layer
        self.target_shape = _collect_input_shape(target_layer)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.target_shape[2]
            width = self.target_shape[3]
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.target_shape[1]
            width = self.target_shape[2]
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs, **kwargs):
        return _resie_image(inputs, self.target_layer, self.target_shape, self.data_format)

    # def get_config(self):
    #     config = {'data_format': self.data_format}
    #     base_config = super(Interpolate, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))
