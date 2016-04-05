import numpy as np
import theano
import os
import time

import numpy as np
import theano
import theano.tensor as T

import pybrain
from pybrain import datasets as ds
from pybrain.tools.shortcuts import buildNetwork


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
	y = [[1],[0],[0],[1],[0],[1],[0],[0],[0],[0]]

	dataset = ds.ClassificationDataSet(30478)
	for x in range(len(arr)):
		dataset.addSample(arr[x].flatten(),y[x])



	return dataset

def build_mlp(input_var=None):
	
	dataset= load_dataset()
	dataset._convertToOneOfMany()
	print dataset



	net = buildNetwork(30478, 100, 1)
	
	

if __name__ == '__main__':
	build_mlp()