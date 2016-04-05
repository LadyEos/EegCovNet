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

	ln = len(x_val)
	print(ln)

	return x_train, y_train, x_val, y_val, x_test, y_test


def iterate_minibatches(inputs, targets, batchsize, shuffle=False): #X_train, y_train, 500, shuffle=True
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main(model='mlp', num_epochs=5):
	x_train, y_train, x_val, y_val, x_test, y_test = load_dataset()

	# Finally, launch the training loop.
	print("Starting training...")
	# We iterate over epochs:
	for epoch in range(num_epochs):
		# In each epoch, we do a full pass over the training data:
		train_err = 0
		train_batches = 0
		start_time = time.time()
		for batch in iterate_minibatches(x_train, y_train, 2, shuffle=True):
			inputs, targets = batch
			#train_err += train_fn(inputs, targets)
			train_batches += 1

		# And a full pass over the validation data:
		val_err = 0
		val_acc = 0
		val_batches = 0
		for batch in iterate_minibatches(x_val, y_val, 1, shuffle=False):
			inputs, targets = batch
			#err, acc = val_fn(inputs, targets)
			val_err += err
			val_acc += acc
			val_batches += 1

		print("Epoch {} of {} took {:.3f}s".format(
		epoch + 1, num_epochs, time.time() - start_time))

	print("Stoping training...")




if __name__ == '__main__':
        main()