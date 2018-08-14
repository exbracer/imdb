#!/usr/bin/env python
import os
import pickle
import subprocess
import sys
import tarfile

import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

if sys.version_info.major == 2:
    # Backward compatibility with python 2
    from six.moves import urllib
    urlretrieve = urllib.request.urlretrieve
else:
    '''
    from urllib.request import urlretrieve
    '''

def shuffle_data(X, y):
    s = np.arange(len(X))
    np.random.shuffle(s)
    X = X[s]
    y = y[s]
    return X, y

def yield_mb(X, y, batchsize=64, shuffle=False):
    assert len(X) == len(y)
    if shuffle:
        X, y = shuffle_data(X, y)
    # Only complete
    for i in range(len(X)//batchsize):
        yield X[i*batchsize:(i+1)*batchsize], y[i*batchsize:(i+1)*batchsize]


def download_imdb(src="https://s3.amazonaws.com/text-datasets/imdb.npz"):
    ''' Load the training and testing data
    '''
    # FLAG:
    print ('Downloading ' + src)
    fname, h = urlretrieve(src, '.delete.me')
    print ('Done.')

    try:
        print('Extracting files...')
        with np.load(fname) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
        print ('Done.')
    finally:
        os.remove(fname)
    return x_train, y_train, x_test, y_test


def read_imdb():
    fname = 'Chainer-Play/imdb/datasets/imdb.npz'
    try:
        print('Extracting files...')
        with np.load(fname) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
            print ('Done.')
    finally:
        # os.remove(fname)
        print ('Load imdb: Done.')
        
    '''        
    x_train = np.load('../datasets/x_train.npy')
    y_train = np.load('../datasets/y_train.npy')
    x_test = np.load('../datasets/x_test.npy')
    y_test = np.load('../datasets/y_test.npy')
    '''
    return x_train, y_train, x_test, y_test


def imdb_for_library(seq_len=100, max_features=20000, one_hot=False, if_local=False):
    ''' Replicates same pre-processing as:
    http://github.com/fchollet/keras/blob/master/keras/datasets.imdb.py

    I just use the implementation of this imdb_for_libaray() function
    directly from the Microsoft Azure implementation
    https://github.com/Azure/learnAnalytics-DeepLearning-Azure/blob/master/Students/9-imdb/common/utils.py
    '''
    # 0 (padding), 1 (start), 2 (OOV)
    START_CHAR = 1
    OOV_CHAR = 2
    INDEX_FROM = 3

    # Raw data (has been encoded into words already)
    if if_local == False:
        x_train, y_train, x_test, y_test = download_imdb()
    else:
        x_train, y_train, x_test, y_test = read_imdb()
    # Combine for processing
    idx = len(x_train)
    _xs = np.concatenate([x_train, x_test])
    # Words will start from INDEX_FROM (shift by 3)
    _xs = [[START_CHAR] + [w + INDEX_FROM for w in x] for x in _xs]
    # Max-features - replace words bigger than index with oov_char
    # E.g. if max_features = 5 then keep 0, 1, 2, 3, 4 i.e. words 3 and 4
    if max_features:
        print("Trimming to {} max-features".format(max_features))
        _xs = [[w if (w < max_features) else OOV_CHAR for w in x] for x in _xs]
    # Pad to same sequences
    print("Padding to length {}".format(seq_len))
    xs = np.zeros((len(_xs), seq_len), dtype=np.int)
    for o_idx, obs in enumerate(_xs):
        # Match keras pre-processing of taking last elements
        obs = obs[-seq_len:]
        for i_idx in range(len(obs)):
            if i_idx < seq_len:
                xs[o_idx][i_idx] = obs[i_idx]
    # One-hot
    if one_hot:
        y_train = np.expand_dims(y_train, axis=-1)
        y_test = np.expand_dims(y_test, axis=-1)
        enc = OneHotEncoder(categorical_features='all')
        fit = enc.fit(y_train)
        y_train = fit.transform(y_train).toarray()
        y_test = fit.transform(y_test).toarray()
    # dtypes
    x_train = np.array(xs[:idx]).astype(np.int32)
    x_test = np.array(xs[idx:]).astype(np.int32)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    return x_train, x_test, y_train, y_test


# For test
'''
MAXLEN=150
MAXFEATURES=20000
imdb_for_library(seq_len=MAXLEN, max_features=MAXFEATURES, one_hot=True, if_local=True)
'''
