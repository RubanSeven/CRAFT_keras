# -*- coding: utf-8 -*-
# @Author: Ruban
# @License: Apache Licence
# @File: train.py

import io
import os
import cv2
import keras
import argparse
import numpy as np
from PIL import Image
from keras import Input
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from module.fake import Fake
from net.vgg16 import VGG16_UNet
from utils.gaussian import GaussianGenerator
from utils.data_util import load_data
from utils.box_util import reorder_points
from utils.img_util import load_sample, img_unnormalize, load_image, img_normalize
from utils.fake_util import crop_image, watershed, find_box, un_warping, cal_confidence, divide_region, \
    enlarge_char_boxes
from module.loss import craft_mse_loss, craft_mae_loss, craft_huber_loss

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.0001)  # initial learning rate
parser.add_argument('--batch_size', type=int, default=5)  # batch size for training
parser.add_argument('--img_size', type=int, default=600)  # batch size for training
parser.add_argument('--max_epochs', type=int, default=800)  # maximum number of epochs
parser.add_argument('--gpu_list', type=str, default='0')  # list of gpus to use
parser.add_argument('--use_fake', type=bool, default=True)  # list of gpus to use
# path to training data
parser.add_argument('--truth_data_path', type=str, default=r'D:\data\synthText\SynthText')
parser.add_argument('--pseudo_data_path', type=str, default=r'D:\data\ICDAR2013')
parser.add_argument('--val_data_path', type=str, default='D:\data\ICDAR2013')  # path to validation data
parser.add_argument('--max_image_size', type=int, default=1280)

FLAGS = parser.parse_args()


class SampleGenerator(keras.callbacks.Callback):
    def __init__(self, base_model, train_sample_lists, train_sample_probs, fakes, img_size, batch_size):
        super().__init__()
        assert len(train_sample_lists) == len(train_sample_probs)
        assert len(train_sample_lists) == len(fakes)
        self.base_model = base_model
        self.train_sample_lists = train_sample_lists
        self.fakes = fakes
        self.train_sample_probs = np.array(train_sample_probs) / np.sum(train_sample_probs)
        self.sample_count_list = [len(sample_list) for sample_list in train_sample_lists]
        self.sample_idx_list = [0] * len(train_sample_lists)
        self.sample_mark_list = list(range(len(train_sample_lists)))
        self.img_size = img_size
        self.batch_size = batch_size

    def get_batch(self, size, is_true=True):
        images = list()
        word_boxes_list = list()
        word_lengths_list = list()
        region_scores = list()
        affinity_scores = list()
        confidence_score_list = list()
        fg_masks = list()
        bg_masks = list()
        gaussian_generator = GaussianGenerator()

        word_count_list = list()
        for i in range(size):
            if is_true:
                sample_mark = np.random.choice(self.sample_mark_list, p=self.train_sample_probs)
            else:
                while 1:
                    sample_mark = np.random.choice(self.sample_mark_list, p=self.train_sample_probs)
                    if self.fakes[sample_mark]:
                        break
            img_path, word_boxes, words, char_boxes_list, confidence_list = \
                self.train_sample_lists[sample_mark][self.sample_idx_list[sample_mark]]
            self.sample_idx_list[sample_mark] += 1
            if self.sample_idx_list[sample_mark] >= self.sample_count_list[sample_mark]:
                self.sample_idx_list[sample_mark] = 0
                np.random.shuffle(self.train_sample_lists[sample_mark])

            img, word_boxes, char_boxes_list, region_box_list, affinity_box_list, img_shape = \
                load_sample(img_path, self.img_size, word_boxes, char_boxes_list)

            images.append(img)

            word_count = min(len(word_boxes), len(words))
            word_boxes = np.array(word_boxes[:word_count], dtype=np.int32) // 2
            word_boxes_list.append(word_boxes)
            word_count_list.append(word_count)

            word_lengths = [len(words[j]) if len(char_boxes_list[j]) == 0 else 0 for j in range(word_count)]
            word_lengths_list.append(word_lengths)

            height, width = img.shape[:2]
            heat_map_size = (height // 2, width // 2)

            mask_shape = (img_shape[1] // 2, img_shape[0] // 2)
            confidence_score = np.ones(heat_map_size, dtype=np.float32)
            for word_box, confidence_value in zip(word_boxes, confidence_list):
                if confidence_value == 1:
                    continue
                tmp_confidence_score = np.zeros(heat_map_size, dtype=np.uint8)
                cv2.fillPoly(tmp_confidence_score, [np.array(word_box)], 1)
                tmp_confidence_score = np.float32(tmp_confidence_score) * confidence_value
                confidence_score = \
                    np.where(tmp_confidence_score > confidence_score, tmp_confidence_score, confidence_score)
            confidence_score_list.append(confidence_score)

            fg_mask = np.zeros(heat_map_size, dtype=np.uint8)
            cv2.fillPoly(fg_mask, [np.array(word_box) for word_box in word_boxes], 1)
            fg_masks.append(fg_mask)
            bg_mask = np.zeros(heat_map_size, dtype=np.float32)
            bg_mask[:mask_shape[0], :mask_shape[1]] = 1
            bg_mask = bg_mask - fg_mask
            bg_mask = np.clip(bg_mask, 0, 1)
            bg_masks.append(bg_mask)

            region_score = gaussian_generator.gen(heat_map_size, np.array(region_box_list) // 2)
            region_scores.append(region_score)

            affinity_score = gaussian_generator.gen(heat_map_size, np.array(affinity_box_list) // 2)
            affinity_scores.append(affinity_score)

        max_word_count = np.max(word_count_list)
        max_word_count = max(1, max_word_count)
        new_word_boxes_list = np.zeros((size, max_word_count, 4, 2), dtype=np.int32)
        new_word_lengths_list = np.zeros((size, max_word_count), dtype=np.int32)
        for i in range(size):
            if word_count_list[i] > 0:
                new_word_boxes_list[i, :word_count_list[i]] = np.array(word_boxes_list[i])
                new_word_lengths_list[i, :word_count_list[i]] = np.array(word_lengths_list[i])

        images = np.array(images)
        region_scores = np.array(region_scores, dtype=np.float32)
        affinity_scores = np.array(affinity_scores, dtype=np.float32)
        confidence_scores = np.array(confidence_score_list, dtype=np.float32)
        fg_masks = np.array(fg_masks, dtype=np.float32)
        bg_masks = np.array(bg_masks, dtype=np.float32)

        inputs = {
            'image': images,
            'word_box': new_word_boxes_list,
            'word_length': new_word_lengths_list,
            'region': region_scores,
            'affinity': affinity_scores,
            'confidence': confidence_scores,
            'fg_mask': fg_masks,
            'bg_mask': bg_masks,
        }

        outputs = {
            'craft': np.zeros([size])
        }

        return inputs, outputs

    def fake_char_boxes(self, src, word_box, word_length):
        img, src_points, crop_points = crop_image(src, word_box, dst_height=64.)
        h, w = img.shape[:2]
        if min(h, w) == 0:
            confidence = 0.5
            region_boxes = divide_region(word_box, word_length)
            region_boxes = [reorder_points(region_box) for region_box in region_boxes]
            return region_boxes, confidence
        img = img_normalize(img)
        # print(img.shape)
        region_score, _ = self.base_model.predict(np.array([img]))
        heat_map = region_score[0] * 255.
        heat_map = heat_map.astype(np.uint8)
        marker_map = watershed(heat_map)
        region_boxes = find_box(marker_map)
        confidence = cal_confidence(region_boxes, word_length)
        if confidence <= 0.5:
            confidence = 0.5
            region_boxes = divide_region(word_box, word_length)
            region_boxes = [reorder_points(region_box) for region_box in region_boxes]
        else:
            region_boxes = np.array(region_boxes) * 2
            region_boxes = enlarge_char_boxes(region_boxes, crop_points)
            region_boxes = [un_warping(region_box, src_points, crop_points) for region_box in region_boxes]
            # print(word_box, region_boxes)

        return region_boxes, confidence

    def init_sample(self, flag=False):
        for sample_mark in self.sample_mark_list:
            if self.fakes[sample_mark]:
                sample_list = self.train_sample_lists[sample_mark]
                new_sample_list = list()

                for sample in sample_list:
                    if len(sample) == 5:
                        img_path, word_boxes, words, _, _ = sample
                    else:
                        img_path, word_boxes, words, _ = sample
                    img = load_image(img_path)
                    char_boxes_list = list()

                    confidence_list = list()
                    for word_box, word in zip(word_boxes, words):
                        char_boxes, confidence = self.fake_char_boxes(img, word_box, len(word))
                        char_boxes_list.append(char_boxes)
                        confidence_list.append(confidence)
                    new_sample_list.append([img_path, word_boxes, words, char_boxes_list, confidence_list])

                self.train_sample_lists[sample_mark] = new_sample_list
            elif flag:
                sample_list = self.train_sample_lists[sample_mark]
                new_sample_list = list()

                for sample in sample_list:
                    if len(sample) == 5:
                        img_path, word_boxes, words, char_boxes_list, _ = sample
                    else:
                        img_path, word_boxes, words, char_boxes_list = sample
                    confidence_list = [1] * len(word_boxes)
                    new_sample_list.append([img_path, word_boxes, words, char_boxes_list, confidence_list])

                self.train_sample_lists[sample_mark] = new_sample_list

    def on_epoch_end(self, epoch, logs=None):
        self.init_sample()

    def on_train_begin(self, logs=None):
        self.init_sample(True)

    def next_train(self):
        while 1:
            ret = self.get_batch(self.batch_size)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.batch_size, False)
            yield ret


def make_image_summary(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    if len(tensor.shape) == 2:
        height, width = tensor.shape
        channel = 1
    else:
        height, width, channel = tensor.shape
        if channel == 1:
            tensor = tensor[:, :, 0]
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


class CraftTensorBoard(TensorBoard):
    def __init__(self, log_dir, write_graph, test_model, callback_model, data_generator):
        self.test_model = test_model
        self.callback_model = callback_model
        self.data_generator = data_generator
        super(CraftTensorBoard, self).__init__(log_dir=log_dir, write_graph=write_graph)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'learning_rate': K.eval(self.model.optimizer.lr)})
        # self.data_generator.init_sample()
        data = next(self.data_generator.next_val())
        images = data[0]['image']
        word_boxes = data[0]['word_box']
        word_lengths = data[0]['word_length']
        target_region = data[0]['region']
        target_affinity = data[0]['affinity']
        confidence_scores = data[0]['confidence']
        region, affinity, region_gt, affinity_gt = self.callback_model.predict([images, word_boxes, word_lengths,
                                                                                target_region, target_affinity,
                                                                                confidence_scores])
        img_summaries = []
        for i in range(3):
            input_image_summary = make_image_summary(img_unnormalize(images[i]))
            pred_region_summary = make_image_summary((region[i] * 255).astype('uint8'))
            pred_affinity_summary = make_image_summary((affinity[i] * 255).astype('uint8'))
            gt_region_summary = make_image_summary((region_gt[i] * 255).astype('uint8'))
            gt_affinity_summary = make_image_summary((affinity_gt[i] * 255).astype('uint8'))
            img_summaries.append(tf.Summary.Value(tag='input_image/%d' % i, image=input_image_summary))
            img_summaries.append(tf.Summary.Value(tag='region_pred/%d' % i, image=pred_region_summary))
            img_summaries.append(tf.Summary.Value(tag='affinity_pred/%d' % i, image=pred_affinity_summary))
            img_summaries.append(tf.Summary.Value(tag='region_gt/%d' % i, image=gt_region_summary))
            img_summaries.append(tf.Summary.Value(tag='affinity_gt/%d' % i, image=gt_affinity_summary))
        tf_summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(tf_summary, epoch + 1)
        super(CraftTensorBoard, self).on_epoch_end(epoch + 1, logs)

        self.test_model.save_weights(r'weights/weight.h5'.format(epoch))


def train():
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    input_image = Input(shape=(None, None, 3), name='image', dtype=tf.float32)
    input_box = Input(shape=(None, 4, 2), name='word_box', dtype=tf.int32)
    input_word_length = Input(shape=(None,), name='word_length', dtype=tf.int32)
    input_region = Input(shape=(None, None), name='region', dtype=tf.float32)
    input_affinity = Input(shape=(None, None), name='affinity', dtype=tf.float32)
    input_confidence = Input(shape=(None, None), name='confidence', dtype=tf.float32)
    input_fg_mask = Input(shape=(None, None), name='fg_mask', dtype=tf.float32)
    input_bg_mask = Input(shape=(None, None), name='bg_mask', dtype=tf.float32)

    region, affinity = VGG16_UNet(input_tensor=input_image, weights='imagenet')

    # if FLAGS.use_fake:
    #     region_gt, affinity_gt, confidence_gt = \
    #         Fake(input_box, input_word_length, input_region, input_affinity, input_confidence, name='fake')(region)
    # else:
    #     region_gt = Lambda(lambda x: x)(input_region)
    #     affinity_gt = Lambda(lambda x: x)(input_affinity)
    #     confidence_gt = Lambda(lambda x: x)(input_confidence)
    region_gt = Lambda(lambda x: x)(input_region)
    affinity_gt = Lambda(lambda x: x)(input_affinity)
    confidence_gt = Lambda(lambda x: x)(input_confidence)

    loss_funs = [craft_mse_loss, craft_mae_loss, craft_huber_loss]

    loss_out = Lambda(loss_funs[2], output_shape=(1,), name='craft')(
        [region_gt, affinity_gt, region, affinity, confidence_gt, input_fg_mask, input_bg_mask])

    model = Model(inputs=[input_image, input_box, input_word_length, input_region,
                          input_affinity, input_confidence, input_fg_mask, input_bg_mask],
                  outputs=loss_out)

    callback_model = Model(inputs=[input_image, input_box, input_word_length, input_region,
                                   input_affinity, input_confidence],
                           outputs=[region, affinity, region_gt, affinity_gt])

    test_model = Model(inputs=[input_image], outputs=[region, affinity])
    test_model.summary()

    weight_path = r'weights/weight.h5'
    if os.path.exists(weight_path):
        test_model.load_weights(weight_path)

    # optimizer = SGD(lr=FLAGS.learning_rate, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    optimizer = Adam(lr=FLAGS.learning_rate)
    model.compile(loss={'craft': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    true_sample_list = load_data(os.path.join(FLAGS.truth_data_path, r'gt.pkl'))

    train_sample_list = true_sample_list

    np.random.shuffle(train_sample_list)

    if FLAGS.use_fake:
        pseudo_sample_list = load_data(os.path.join(FLAGS.pseudo_data_path, r'gt.pkl'))
        np.random.shuffle(pseudo_sample_list)
        train_generator = SampleGenerator(test_model, [train_sample_list, pseudo_sample_list], [5, 1], [False, True],
                                          FLAGS.img_size, FLAGS.batch_size)
        # tensor_board_data_generator = SampleGenerator(test_model, [pseudo_sample_list], [1], [True],
        #                                               FLAGS.img_size, FLAGS.batch_size)
    else:
        train_generator = SampleGenerator(test_model, [train_sample_list], [1], [False],
                                          FLAGS.img_size, FLAGS.batch_size)
        # tensor_board_data_generator = SampleGenerator(test_model, [train_sample_list], [1], [False],
        #                                               FLAGS.img_size, FLAGS.batch_size)

    # train_generator.init_sample(True)

    # val_pkl_path = os.path.join(FLAGS.val_data_path, r'gt.pkl')
    # if os.path.exists(val_pkl_path):
    #     val_sample_list = load_data(val_pkl_path)

    steps_per_epoch = 1000

    tensor_board = CraftTensorBoard(log_dir=r'logs',
                                    write_graph=False,
                                    test_model=test_model,
                                    callback_model=callback_model,
                                    data_generator=train_generator,
                                    )

    model.fit_generator(generator=train_generator.next_train(),
                        steps_per_epoch=steps_per_epoch,
                        initial_epoch=0,
                        epochs=FLAGS.max_epochs,
                        callbacks=[train_generator, tensor_board]
                        )


if __name__ == '__main__':
    train()
