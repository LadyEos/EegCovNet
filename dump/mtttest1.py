import os
import time

import numpy as np
from modified_theano_tutorials import DBN as dbn, rbm,logistics_cg as logistics
import theano

def test_dbn(dataset):
	DBN.test_DBN(pretraining_epochs=1, training_epochs=1, batch_size=300,dataset=dataset,n_ins=2177)

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

	#y = [1,2,3,4,5,6,7,8,9,10]
	#y = [[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]]
	y = np.ones_like(arr)


	return [arr,y]

if __name__ == '__main__':
  test_dbn(load_dataset())

