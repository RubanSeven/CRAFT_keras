# -*- coding: utf-8 -*-
# @Author: Ruban
# @License: Apache Licence
# @File: icdar2013_convert.py

import os
import re
import codecs
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=r'D:\data\ICDAR2013')

FLAGS = parser.parse_args()


class ICDAR2013Convertor:
    def __init__(self, img_root, txt_root):
        super(ICDAR2013Convertor, self).__init__()
        self.img_root = img_root
        self.txt_root = txt_root

    def convert_to_craft(self):
        img_name_list = os.listdir(self.img_root)
        sample_list = list()
        for img_name in img_name_list:
            txt_name = 'gt_' + img_name[:-len(img_name.split('.')[-1])] + 'txt'
            txt_path = os.path.join(self.txt_root, txt_name)
            img_path = os.path.join(self.img_root, img_name)
            if os.path.exists(txt_path):
                word_boxes = list()
                char_boxes_list = list()
                words = list()
                with codecs.open(txt_path, 'rb', encoding='utf-8') as txt_file:
                    lines = txt_file.read().splitlines()
                    for line in lines:
                        infos = re.split(',? ', line)

                        word = infos[4]
                        word = re.sub('^"', '', word)
                        word = re.sub('"$', '', word)
                        if '\\' in word:
                            print(word)
                        words.append(word)

                        char_boxes_list.append([])

                        left, top, right, bottom = [round(float(p)) for p in infos[:4]]
                        word_box = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
                        word_boxes.append(word_box)

                sample_list.append([img_path, word_boxes, words, char_boxes_list])

        return sample_list


if __name__ == '__main__':
    import pickle

    image_dir = os.path.join(FLAGS.data_dir, r'Challenge2_Training_Task12_Images')
    txt_dir = os.path.join(FLAGS.data_dir, r'Challenge2_Training_Task1_GT')
    pkl_path = os.path.join(FLAGS.data_dir, r'gt.pkl')

    icdar_2013_convertor = ICDAR2013Convertor(img_root=image_dir,
                                              txt_root=txt_dir)
    craft_sample_list = icdar_2013_convertor.convert_to_craft()
    with open(pkl_path, 'wb') as pkl_file:
        pickle.dump(craft_sample_list, pkl_file)

    image_dir = os.path.join(FLAGS.data_dir, r'Challenge2_Test_Task12_Images')
    txt_dir = os.path.join(FLAGS.data_dir, r'Challenge2_Test_Task1_GT')
    pkl_path = os.path.join(FLAGS.data_dir, r'test_gt.pkl')

    icdar_2013_convertor = ICDAR2013Convertor(img_root=image_dir,
                                              txt_root=txt_dir)
    craft_sample_list = icdar_2013_convertor.convert_to_craft()
    with open(pkl_path, 'wb') as pkl_file:
        pickle.dump(craft_sample_list, pkl_file)
