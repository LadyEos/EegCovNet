import numpy as np
import theano
import os
import time
from glob import glob
import pandas as pd
import math

import numpy as np
import theano
import theano.tensor as T

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

def doFile():
	trials = np.array(['01','02','03','04','05','06','07','08','09','10'])
	e_raw = []
	raw = []

	for trial in trials:
		raw_event_files = glob('txt/0-%s_event.csv' % (trial))
		raw_trial_files = glob('txt/0-%s-short.txt' % (trial))
	
		# raw_event_files = glob('txt/0-01_event.csv')
		# raw_trail_files = glob('txt/0-01-short.txt')
		
		e = []
		dh = []

		for raw_event_file in raw_event_files:
			events = pd.read_csv(raw_event_file, sep='\t', header=None)
			events = events.drop(events.columns[[0, 3]], axis=1)
			events = events.ix[1:]
			e = events.as_matrix()


		for raw_trial_file in raw_trial_files:
			data = pd.read_csv(raw_trial_file,sep='\t')
			data = data.drop('NAME',axis=1)
			data = data.drop(data.columns[[2177]], axis=1)
			dh = list(data.columns.values)

		table = np.zeros((len(dh),2))

		for x in range(len(table)):
			for y in range(len(table[x])):
				if(y == 0):
					table[x][y] = dh[x]

		length = len(e)
		counter = 0
		flag = False
		for t in range(len(table)):
			if(t+1 < len(table)):
				if(math.floor(table[t][0]) <= float(e[counter][0])):
					if(math.floor(table[t+1][0]) > float(e[counter][0])):
						if(int(e[counter][1]) == 50):
							flag = True
						else:
							flag = False
						counter += 1
			table[t][1] = flag
			if(counter >= len(e)):
				break

		file = open("events_"+trial+".txt", "w")
		for t in table:
			file.write(str(t[0])+"\t"+str(t[1])+"\n")
		file.close()


if __name__ == '__main__':
	doFile()