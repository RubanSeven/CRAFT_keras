# -*- coding: utf-8 -*-
# @Author: Ruban
# @License: Apache Licence
# @File: conv.py

from keras.layers import Conv2D, BatchNormalization, Activation


def up_conv(input_tensor, filters):
    x = Conv2D(filters[0], kernel_size=1)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters[1], kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def conv_cls(input_tensor, num_class):
    """
    :param input_tensor: 4D tensor.
    :return: 4D tensor. Region score and Affinity score.
    """
    x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(input_tensor)
    x = Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(16, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(16, kernel_size=1, padding='same', activation='relu')(x)
    x = Conv2D(16, kernel_size=num_class, padding='same', activation='sigmoid')(x)
    return x
