# -*- coding: utf-8 -*-
# @Author: Ruban
# @License: Apache Licence
# @File: vgg16.py

from keras import Input, Model
from module.conv import up_conv, conv_cls
from module.interpolate import Interpolate
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, concatenate, MaxPooling2D, Lambda


def VGG16_UNet(weights=None, input_tensor=None, pooling=None):
    vgg16 = VGG16(include_top=False, weights=weights, input_tensor=input_tensor, pooling=pooling)

    source = vgg16.get_layer('block5_conv3').output
    x = MaxPooling2D(3, strides=1, padding='same', name='block5_pool')(source)
    x = Conv2D(1024, kernel_size=3, padding='same', dilation_rate=6)(x)
    x = Conv2D(1024, kernel_size=1)(x)

    x = Interpolate(target_layer=source, name='resize_1')(x)
    x = concatenate([x, source])
    x = up_conv(x, [512, 256])

    source = vgg16.get_layer('block4_conv3').output
    x = Interpolate(target_layer=source, name='resize_2')(x)
    x = concatenate([x, source])
    x = up_conv(x, [256, 128])

    source = vgg16.get_layer('block3_conv3').output
    x = Interpolate(target_layer=source, name='resize_3')(x)
    x = concatenate([x, source])
    x = up_conv(x, [128, 64])

    source = vgg16.get_layer('block2_conv2').output
    x = Interpolate(target_layer=source, name='resize_4')(x)
    x = concatenate([x, source])
    feature = up_conv(x, [64, 32])

    x = conv_cls(feature, 2)

    region_score = Lambda(lambda layer: layer[:, :, :, 0])(x)
    affinity_score = Lambda(lambda layer: layer[:, :, :, 1])(x)

    return region_score, affinity_score


if __name__ == '__main__':
    input_image = Input(shape=(512, 512, 3))
    region, affinity = VGG16_UNet(input_tensor=input_image, weights=None)
    model = Model(input_image, [region, affinity], name='vgg16_unet')
    model.summary()
