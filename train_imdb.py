#!/usr/bin/env python
""" Sample script of traing RNN(GRU) on IMDB - Sentiment
Analysis task

"""

import numpy as np
import os
import sys
import math

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from chainer import optimizers
#from chainer import cuda

from datasets.imdb import *
from params_gru import *

# Performance Improvement
# 1. Auto-tune
# This is recommanded by ilkarman

# Force one-gpu
os.environ["CUDA_VISIBLE_DEVICE"] = "0"

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Chainer: ", chainer.__version__)
#print("CuPy: ", chainer.cuda.cupy.__version__)
print("Numpy: ", np.__version__)
#print("GPU: ", get_gpu_name())
#print(get_cuda_version())
#print("CuDNN Version ", get_cudnn_version())

# Defination of a GRU for IMDB sentiment analysis
class SymbolModule(chainer.Chain):

	def __init__(self, n_vocab, n_units):
		super(SymbolModule, self).__init__()
		with self.init_scope():
			'''
			TODO
			'''
	def __call__(self, x):
		'''
		TODO
		'''

def init_model():
	'''
	TODO
	'''

# Data into format for library
#%%time
x_train, x_test, y_train, y_test = imdb_for_library(seq_len=MAXLEN, max_features=MAXFEATURES, if_local=True)
x_train = x_train.astype(np.int64)
x_test = x_test.astype(np.int64)
y_train = y_train.astype(np.int64)
y_test = y_train.astype(np.int64)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)	

#%%time
sym = SymbolModule()
#sym.cuda() # CUDA!

#%%time
optimizer, criterion = init_model(sym)
