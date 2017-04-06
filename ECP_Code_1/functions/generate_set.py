# -*- coding: utf-8 -*-
"""
@author: Colin Pierre
Generation of data from input to cookies matrix X and labels vector y
"""


import pickle
import random
import numpy as np

def main():
    X, y, l = get_dataset()
    #Set utilisable!

def get_dataset():
    def expand(path, en_vect):
        vectors = []
        main_dic = pickle.load(open(path, 'rb'))
        for _, dict_vec in main_dic.items():
            v = [0]*len_vec
            for feature, value in dict_vec.items():
                v[feature] = value
            vectors.append(v)
        return vectors

    len_vec = pickle.load(open('len_dic.p', 'rb'))
    print 'vectors length: %d' % len_vec
    vectors_pos = expand('pos_train.p', len_vec)
    print 'nb ex csp+: %d' % len(vectors_pos)
    vectors_neg = expand('neg_train.p', len_vec)
    print 'nb ex csp-: %d' % len(vectors_neg)

    vector = vectors_pos + vectors_neg
    target = [1]*len(vectors_pos) + [0]*len(vectors_neg)
    z = zip(vector, target)
    random.shuffle(z)
    x, y = zip(*z)
    return np.array(x), np.array(y), len_vec

if __name__ == '__main__':
    main()
