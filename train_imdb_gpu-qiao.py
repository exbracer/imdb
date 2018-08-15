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
from chainer import Variable

from chainer import optimizers
from chainer import cuda

from common.utils import *
from common.params_gru import *

# Performance Improvement
# 1. Auto-tune
# This is recommanded by ilkarman

# Force one-gpu
os.environ["CUDA_VISIBLE_DEVICE"] = "0"

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("Chainer: ", chainer.__version__)
print("CuPy: ", chainer.cuda.cupy.__version__)
print("Numpy: ", np.__version__)
print("GPU: ", get_gpu_name())
print("CuDNN Version ", get_cudnn_version())

# Defination of a GRU for IMDB sentiment analysis
class SymbolModule(chainer.Chain):

	def __init__(self, maxf=MAXFEATURES, edim=EMBEDSIZE, nhid=NUMHIDDEN):
		super(SymbolModule, self).__init__()
		with self.init_scope():
			'''
			TODO
			'''
			self.embedding = L.EmbedID(in_size=maxf, out_size=edim)
			self.gru = L.NStepGRU(n_layers=1, in_size=edim, out_size=nhid, dropout=0)
			self.l_out = L.Linear(in_size=nhid*1, out_size=2)
	def __call__(self, x, nhid=NUMHIDDEN, batchsize=BATCHSIZE):
		'''
		TODO
		'''
		_x = self.embedding(x)
		_xs = [F.squeeze(xe, 0) for xe in F.split_axis(_x, batchsize, 0)]
		#_xs = F.split_axis(_x, batchsize, 0)
		#print len(_xs)
		#print _xs[0].shape
		_hy, _ys = self.gru(hx=None, xs=_xs)
		#print _hy[0].shape
		#_hy = _hy[:,-1,:].squeeze()
		_h = self.l_out(_hy[0])
		#print _h.shape
		return _h

def init_model(m, lr=LR, b1=BETA_1, b2=BETA_2, eps=EPS):
	optimizer = optimizers.Adam(alpha=lr, beta1=b1, beta2=b2, eps=eps)
	optimizer.setup(m)
	return optimizer


# Data into format for library
# %%time
x_train, x_test, y_train, y_test = imdb_for_library(seq_len=MAXLEN, max_features=MAXFEATURES, if_local=True)
x_train = x_train.astype(np.int64)
x_test = x_test.astype(np.int64)
y_train = y_train.astype(np.int64)
y_test = y_train.astype(np.int64)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)	

# %%time
# create symbol
sym = SymbolModule()
chainer.cuda.get_device(0).use() # Make a specified GPU current
sym.to_gpu() # Copy the model to the GPU

# %%time
optimizer = init_model(sym)


# %%time
# Main training loop: 
for j in range(EPOCHS):
	for data, target in yield_mb(x_train, y_train, BATCHSIZE, shuffle=True):
		# Get samples to gpu
		data = cuda.to_gpu(data)
		target = cuda.to_gpu(target)
		# Forwards propagation
		output = sym(data)
		# Loss
		loss = F.softmax_cross_entropy(output, target)
		sym.cleargrads()
		# Backwards propagation
		loss.backward()
		# Update
		optimizer.update()
	# Log
	print (j) 

# %%time
# Main evaluation loop:
n_samples = (y_test.shape[0]//BATCHSIZE)*BATCHSIZE
y_guess = np.zeros(n_samples, dtype=np.int)
y_truth = y_test[:n_samples]
c = 0
with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
	for data, target in yield_mb(x_test, y_test, BATCHSIZE):
		# Forward propagations
		pred = cuda.to_cpu(sym(cuda.to_gpu(data)).data.argmax(-1))
		#pred = sym(data).data.argmax(-1) # CPU-only version
		# Collect results
		y_guess[c*BATCHSIZE:(c+1)*BATCHSIZE] = pred
		c += 1

# Print the accuracy
print("Accuracy: ", sum(y_guess == y_truth)/len(y_guess))
