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

def main():

	dataAll = load_dataset()
	datas = dataAll[0]
	data = dataAll[0:3:1]
	print(data)
	data = data / data.std(axis=1)[:, np.newaxis]
	print("-------------------")
	print(data)
	data = data - data.mean(axis=1)[:, np.newaxis]
	print("*******************")
	print(data)


if __name__ == '__main__':
	main()