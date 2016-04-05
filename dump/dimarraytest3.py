from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T


def load_dataset():
    
	# Load the files
	temp = []
	path = 'txt/'
	listing = os.listdir(path)
	for infile in listing:
		arrays = np.genfromtxt(path+infile)
		trim = arrays[:,1:]
		trim = trim[1:15]
		temp.append(trim)

	arr = np.zeros((10,14,2177))


	for x in range(len(temp)):
		arr[x] = temp[x]

	return arr

def scale(data):
	# return preprocessing.scale(data, axis=1)
	data = data / data.std(axis=1)[:, np.newaxis]
	data = data - data.mean(axis=1)[:, np.newaxis]

	# print data.std(axis=1).sum()
	# print np.ones((data.shape[0]), dtype='float')
	# assert np.array_equal(data.std(axis=1), np.ones((data.shape[0]), dtype='float'))
	# assert np.array_equal(data.mean(axis=1), np.zeros(data.shape[0]))
	return data

def shuffle(*args):
	shuffled = shuffleList(*args)
	f = lambda x: np.array(x)
	return tuple(map(f, shuffled))


# Returns lists
def shuffleList(*args):
	lenght = len(args[0])

	# Assert they all have the same size
	assert np.array_equal(np.array(map(len, args)), np.ones(len(args)) * lenght)

	indexShuffle = np.random.permutation(lenght)

	f = lambda x: [x[i] for i in indexShuffle]
	return tuple(map(f, args))

def main():

	data = load_dataset()

	print("scaling input data")
	data = scale(data)
	labels = [1,2,3,4,5,6,7,8,9,10]
	# labels = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
	# 			[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
	# 			[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
	# 			[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
	# 			[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
	# 			[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
	# 			[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
	# 			[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
	# 			[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
	# 			[1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
	data, labels = shuffle(data, labels)
	print(data)
	print(labels)





if __name__ == '__main__':
	main()