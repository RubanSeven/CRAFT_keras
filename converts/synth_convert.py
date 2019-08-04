# -*- coding: utf-8 -*-
# @Author: Ruban
# @License: Apache Licence
# @File: synth_convert.py

import os
import re
import numpy as np
import scipy.io as sio


class SynthTextConvertor:
    def __init__(self, mat_path, image_root):
        super(SynthTextConvertor, self).__init__()
        self.mat_path = mat_path
        self.image_root = image_root
        self.image_name_list, self.word_boxes_list, self.char_boxes_list, self.texts_list = self.__load_mat()

    def __load_mat(self):
        data = sio.loadmat(self.mat_path)
        image_name_list = data['imnames'][0]
        word_boxes_list = data['wordBB'][0]
        char_boxes_list = data['charBB'][0]
        texts_list = data['txt'][0]

        return image_name_list, word_boxes_list, char_boxes_list, texts_list

    @staticmethod
    def split_text(texts):
        split_texts = list()
        for text in texts:
            text = re.sub(' ', '', text)
            split_texts += text.split()
        return split_texts

    @staticmethod
    def swap_box_axes(boxes):
        if len(boxes.shape) == 2 and boxes.shape[0] == 2 and boxes.shape[1] == 4:
            # (2, 4) -> (1, 4, 2)
            boxes = np.array([np.swapaxes(boxes, axis1=0, axis2=1)])
        else:
            # (2, 4, n) -> (n, 4, 2)
            boxes = np.swapaxes(boxes, axis1=0, axis2=2)
        return boxes

    def convert_to_craft(self):
        sample_list = list()
        for image_name, word_boxes, char_boxes, texts in zip(self.image_name_list, self.word_boxes_list,
                                                             self.char_boxes_list, self.texts_list):
            word_boxes = self.swap_box_axes(word_boxes)
            char_boxes = self.swap_box_axes(char_boxes)
            texts = self.split_text(texts)
            tmp_char_boxes_list = list()
            char_index = 0
            for text in texts:
                char_count = len(text)
                tmp_char_boxes_list.append(char_boxes[char_index:char_index + char_count])
                char_index += char_count
            image_path = os.path.join(self.image_root, image_name[0])
            sample_list.append([image_path, word_boxes, texts, tmp_char_boxes_list])

        return sample_list


if __name__ == '__main__':
    import pickle

    synth_text_convertor = SynthTextConvertor(mat_path=r'D:\data\synthText\SynthText\gt.mat',
                                              image_root=r'D:\data\synthText\SynthText')
    craft_sample_list = synth_text_convertor.convert_to_craft()
    np.random.shuffle(craft_sample_list)
    with open(r'D:\data\synthText\SynthText\gt.pkl', 'wb') as pkl_file:
        pickle.dump(craft_sample_list[:int(len(craft_sample_list) / 80)], pkl_file)
