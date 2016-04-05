import numpy as np
import pandas as pd
from glob import glob

def load_dataset_X(SIZE):

	# Load the files
	raw = []
	y_raw = []

	trials = np.arange(1,101)

	for trial in trials:
		raw_trial_files = glob('data2/trial/0-%s.txt' % (trial))
		raw_event_files = glob('data2/event/events_0-%s.txt' % (trial))
		
		for raw_event_file in raw_event_files:
			events = pd.read_csv(raw_event_file, sep='\t', header=None)
			#events = events.ix[1:]
			events = events.drop(events.columns[[0]],axis=1)
			events = events.as_matrix()
			y_raw.append(events)

		for raw_trial_file in raw_trial_files:
			data = pd.read_csv(raw_trial_file,sep='\t')
			data = data.drop('NAME',axis=1)
			data = data.drop(data.columns[[SIZE]], axis=1)
			#data = data.ix[1:]
			data = data.as_matrix()
			raw.append(data.transpose())
		

	X = np.zeros_like(raw[0][0])
	for t in raw:
		X =  np.vstack((X,t))
	X = X[1:]
	print X.shape

	y = np.zeros_like(y_raw[0][0])
	for t in y_raw:
		y =  np.vstack((y,t))
	y = y[1:]
	print y.shape

	return X, y


def load_testdata(SIZE):
	print("Reading test data .... ")
	test_dict = dict()
	test_total = 0
	ids_tot = []
	test = []
	idt = []
	idx = []

	trials = np.arange(1,21)

	for trial in trials:
		raw_trial_files = glob('data/txtshort/0-%s-short.txt' % (trial))

		for raw_trial_file in raw_trial_files:

			data = pd.read_csv(raw_trial_file,sep='\t',header=0)
			data = data.drop('NAME',axis=1)
			data = data.drop(data.columns[[SIZE]], axis=1) #remove Unnamed: 2178
			
			idt = np.array(list(data.columns.values)).transpose()

			data = data.as_matrix()
			data = data.transpose()
			test.append(data)

			data_size = len(data)
			data_name = 'subj{0}_series{1}'.format('0', trial)
			test_dict[data_name] = data_size

			test_total += data_size

			for x in idt:
				new_name = data_name + '_' + x
				idx.append(new_name)

	if idx:
		ids = idx
		ids_tot = ids

	X = np.zeros_like(test[0][0])
	for t in test:
		X =  np.vstack((X,t))
	X = X[1:]


	
	return X, ids_tot,test_dict,test_total

def build_dbn():
	cols = ['Start']
	SIZE = 1665

	# X, y = load_dataset(SIZE)
	X_test, ids_tot, test_dict, test_total = load_testdata(SIZE)

	print "finish"

if __name__ == '__main__':
	build_dbn()