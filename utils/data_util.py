# -*- coding: utf-8 -*-
# @Author: Ruban
# @License: Apache Licence
# @File: data_util.py

import pickle


def load_data(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)

    return data
