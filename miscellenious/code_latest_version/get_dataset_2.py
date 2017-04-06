# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 11:17:41 2016

@author: paulvernhet
""" 
"get dataset = Import des données"

import pickle
import numpy as np
import random

def get_dataset():
    
    def expand(path, len_vect):
        vectors = []
        main_dic = pickle.load(open(path, 'rb'))
        for _, dict_vec in main_dic.items():
            v = [0]*len_vec
            for feature, value in dict_vec.items():
                v[feature] = value
            vectors.append(v)
        return vectors

    len_vec = pickle.load(open('lookalike_csp_len_dic.p', 'rb'))
    print('vectors length: %d' % len_vec)
    vectors_pos = expand('lookalike_csp_vector_pos_train.p', len_vec)
    print('nb ex csp+: %d' % len(vectors_pos))
    vectors_neg = expand('lookalike_csp_vector_neg_train.p', len_vec)
    print('nb ex csp-: %d' % len(vectors_neg))

    vector = vectors_pos + vectors_neg 
    target = [1]*len(vectors_pos) + [0]*len(vectors_neg)
    z = list(zip(vector, target))
    random.shuffle(z)
    x, y = zip(*z)
    return np.array(x), np.array(y), len_vec