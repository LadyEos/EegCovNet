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

	currentData = load_dataset()
	for x in range(0,len(currentData))
        print(x)





if __name__ == '__main__':
	main()