# -*- coding: utf-8 -*-
# @Author: Ruban
# @License: Apache Licence
# @File: test.py

import cv2
import os
import time
import argparse
import numpy as np
import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from net.vgg16 import VGG16_UNet
from collections import OrderedDict
from utils.file_util import list_files, saveResult
from utils.inference_util import getDetBoxes, adjustResultCoordinates
from utils.img_util import load_image, img_resize, img_normalize, to_heat_map


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/weight.h5', type=str, help='pretrained model')
parser.add_argument('--gpu_list', type=str, default='0', help='list of gpu to use')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1., type=float, help='image magnification ratio')
parser.add_argument('--show_time', default=True, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default=r'D:\data\ICDAR2013\Challenge2_Test_Task12_Images',
                    type=str, help='folder path to input images')

FLAGS = parser.parse_args()

result_folder = 'results/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)


def predict(model, image, text_threshold, link_threshold, low_text):
    t0 = time.time()

    # resize
    h, w = image.shape[:2]
    mag_ratio = 600 / max(h, w)
    # img_resized, target_ratio = img_resize(image, FLAGS.mag_ratio, FLAGS.canvas_size, interpolation=cv2.INTER_LINEAR)
    img_resized, target_ratio = img_resize(image, mag_ratio, FLAGS.canvas_size, interpolation=cv2.INTER_LINEAR)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = img_normalize(img_resized)

    # make score and link map
    score_text, score_link = model.predict(np.array([x]))
    score_text = score_text[0]
    score_link = score_link[0]

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text)
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    white_img = np.ones((render_img.shape[0], 10, 3), dtype=np.uint8) * 255
    ret_score_text = np.hstack((to_heat_map(render_img), white_img, to_heat_map(score_link)))
    # ret_score_text = to_heat_map(render_img)

    if FLAGS.show_time:
        print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, ret_score_text


def test():
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    """ Load model """
    input_image = Input(shape=(None, None, 3), name='image', dtype=tf.float32)
    region, affinity = VGG16_UNet(input_tensor=input_image, weights=None)
    model = Model(inputs=[input_image], outputs=[region, affinity])
    model.load_weights(FLAGS.trained_model)

    """ For test images in a folder """
    image_list, _, _ = list_files(FLAGS.test_folder)

    t = time.time()

    """ Test images """
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
        image = load_image(image_path)
        start_time = time.time()
        bboxes, score_text = predict(model, image, FLAGS.text_threshold, FLAGS.link_threshold, FLAGS.low_text)
        print(time.time() * 1000 - start_time * 1000)

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        saveResult(image_path, image[:, :, ::-1], bboxes, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))


if __name__ == '__main__':
    test()
