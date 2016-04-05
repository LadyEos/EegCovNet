import numpy as np
import theano
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
from nolearn.lasagne import NeuralNet
from lasagne import layers
from lasagne.updates import nesterov_momentum

# ################## LOAD DATASET ##################
# from data/txt
# and load it into numpy arrays.

def load_dataset():

	# Load the files
	temp = []
	path = 'data/txtshort/'
	listing = os.listdir(path)
	for infile in listing:
		arrays = np.genfromtxt(path+infile)
		trim = arrays[:,1:]
		trim = trim[1:15]
		temp.append(trim)

	arr = np.zeros((10,14,2177))


	for xr in range(len(temp)):
		arr[xr] = temp[xr]

	#y = np.ones(10)
	y = [1,0,0,1,0,1,0,0,0,0]


	return arr, y

def build_mlp(input_var=None):
	net1 = NeuralNet(
	layers=[  # three layers: one hidden layer
			('input', layers.InputLayer),
			('hidden1', layers.DenseLayer),
			('hidden2', layers.DenseLayer),
			('output', layers.DenseLayer),
		],
	# layer parameters:
	input_shape=(None, 14, 2177),  #  14 x 2177 input pixels per batch
	hidden1_num_units=100,  # number of units in hidden layer
	hidden2_num_units=100,
	output_nonlinearity=lasagne.nonlinearities.softmax,  # output layer uses identity function
	output_num_units=2,  # 2 target values

	# optimization method:
	update=nesterov_momentum,
	update_learning_rate=0.01,
	update_momentum=0.9,

	#regression=False,  # flag to indicate we're dealing with regression problem
	max_epochs=500,  # we want to train this many epochs
	verbose=1,
	)

	X, y = load_dataset()
	y = np.asanyarray(y,np.int32)
	print(X.shape)
	print(y.shape)
	net1.fit(X, y)

if __name__ == '__main__':
	build_mlp()