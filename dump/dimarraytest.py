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


	x_train = arr[:7]
	y_train = arr[:7]

	x_val = arr[7:9]
	y_val = arr[7:9]
	x_test = arr[-1] 
	y_test = arr[-1]

	return arr

def main():

	dataAll = load_dataset()
	data = dataAll
	mins = data.min(axis=2)
	maxs = data.max(axis=2)
	# print("min")
	# print(mins)
	# print("max")
	# print(maxs)

	b = np.all(mins >= 0.0)
	c = np.all(maxs < 1.0 + 1e-8)

	d = np.all(mins) >=0.0
	f = np.all(maxs) < 1.0 + 1e-8



	#assert np.all(mins) >=0.0 and np.all(maxs) < 1.0 + 1e-8

	print("b")
	print(b)
	print("c")
	print(c)
	print("d")
	print(d)
	print("f")
	print(f)
	#assert np.all(mins >=0.0) and np.all(maxs < 1.0 + 1e-8)

	assert d and f


if __name__ == '__main__':
	main()