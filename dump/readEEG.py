import cPickle as pickle
import pandas as pd
from glob import glob
from datetime import datetime
from time import gmtime, strftime
import numpy as np

def load_dataset():

	# Load the files
	raw = []
	y_raw = []

	trials = np.arange(1,3)

	# for trial in trials:
	raw_trial_files = sorted(glob('data2/ICA/trial/0-*.txt'))
	raw_event_files = sorted(glob('data2/ICA/event/events_0-*.txt'))

	# print raw_trial_files
	# print raw_event_files
	
	for raw_event_file in raw_event_files:
		print raw_event_file
		events = pd.read_csv(raw_event_file, sep='\t', header=None)
		#events = events.ix[1:]
		events = events.drop(events.columns[[0]],axis=1)
		events = events.as_matrix()
		y_raw.append(events)

	for raw_trial_file in raw_trial_files:
		print raw_trial_file
		data = pd.read_csv(raw_trial_file,sep='\t')
		data = data.drop('NAME',axis=1)
		last_col = len(data.columns)
		data = data.drop(data.columns[[last_col-1]], axis=1)
		#data = data.ix[1:]
		data = data.as_matrix()
		raw.append(data.transpose())
		
	print '#####################################################################'
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

	# return None , None

	return X, y

def data_preprocess(X):
    # normalize data
    mean = X.mean(axis=0)
    X -= mean
    std = X.std(axis=0)
    X /= std
    X_prep = X
    return X_prep

def load_testdata():
	print("Reading test data .... ")
	test_dict = dict()
	test_total = 0
	ids_tot = []
	test = []
	idt = []
	idx = []

	trials = np.arange(1,21)


	raw_trial_files = sorted(glob('data2/ICA2/test/trial/1-*.txt'))

	counter = 1

	for raw_trial_file in raw_trial_files:

		data = pd.read_csv(raw_trial_file,sep='\t',header=0)
		data = data.drop('NAME',axis=1)
		last_col = len(data.columns)
		data = data.drop(data.columns[[last_col-1]], axis=1) #remove Unnamed: 2178
		
		idt = np.array(list(data.columns.values)).transpose()

		data = data.as_matrix()
		data = data.transpose()
		test.append(data)

		data_size = len(data)
		data_name = 'subj{0}_test{1}'.format('1', counter)
		test_dict[data_name] = data_size

		test_total += data_size
		counter += 1

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


def pickleit():
	print 'Start'
	# X, y = load_dataset()
	# trialdata = {'X' : X, 'y':y}
	X_test, ids_tot, test_dict, test_total = load_testdata()
	testdata = {'X_test':X_test, 'ids_tot' : ids_tot, 'test_dict': test_dict, 'test_total': test_total}
	print 'store pickle'

	# with open('datapickled/traildata.pickle', 'wb') as f:
	# 	pickle.dump(trialdata, f, -1)

	with open('datapickled/testdata2.pickle', 'wb') as f:
		pickle.dump(testdata, f, -1)
	print 'finish'

if __name__ == '__main__':
	pickleit()