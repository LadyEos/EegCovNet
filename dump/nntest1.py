import numpy as np
import pandas as pd
from glob import glob
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import theano
from theano.tensor.nnet import sigmoid
from sklearn.preprocessing import StandardScaler
from lasagne.objectives import aggregate, binary_crossentropy
from matplotlib import pyplot

def load_dataset(SIZE):

	# Load the files
	raw = []
	y_raw = []

	trials = np.arange(1,101)

	# for trial in trials:
	raw_trial_files = glob('data2/trial/0-*.txt')
	raw_event_files = glob('data2/event2/events_0-*.txt')
	
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

def loss(x, t):
    return aggregate(binary_crossentropy(x, t))

def load_testdata(SIZE):
	print("Reading test data .... ")
	test_dict = dict()
	test_total = 0
	ids_tot = []
	test = []
	idt = []
	idx = []

	trials = np.arange(1,21)


	raw_trial_files = glob('data2/test/1-*.txt')

	counter = 1

	for raw_trial_file in raw_trial_files:

		data = pd.read_csv(raw_trial_file,sep='\t',header=0)
		data = data.drop('NAME',axis=1)
		data = data.drop(data.columns[[SIZE]], axis=1) #remove Unnamed: 2178
		
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

def data_preprocess(X):
    # normalize data
    mean = X.mean(axis=0)
    X -= mean
    std = X.std(axis=0)
    X /= std
    X_prep = X
    return X_prep


def build_dbn():
	cols = ['Start']
	SIZE = 1665

	t = 1


	channels = 14
	batch_size = None  #None = arbitary batch size
	hidden_layer_size = 100  #change to 1024
	N_EVENTS = 2
	max_epochs = 500
	NO_TIME_POINTS = 10


	ids_tot = []
	pred_tot = []
	test_dict = dict()

	test_total = 0


	X, y = load_dataset(SIZE)
	# X_test, ids_tot, test_dict, test_total = load_testdata(SIZE)
	
	

	net = NeuralNet(
		layers=[
			('input', layers.InputLayer),
			# ('dropout1', layers.DropoutLayer),
			# ('conv1', layers.Conv1DLayer),
			# ('conv2', layers.Conv1DLayer),
			# ('pool1', layers.MaxPool1DLayer),
			# ('dropout2', layers.DropoutLayer),
			('hidden4', layers.DenseLayer),
			# ('dropout3', layers.DropoutLayer),
			('hidden5', layers.DenseLayer),
			# ('dropout4', layers.DropoutLayer),
			('output', layers.DenseLayer),
		],
		input_shape=(None, channels, NO_TIME_POINTS),
		# dropout1_p=0.5,
		# conv1_num_filters=4, conv1_filter_size=1,
		# conv2_num_filters=8, conv2_filter_size=4, pool1_pool_size=4,
		# dropout2_p=0.5, 
		# dropout3_p=0.5, 
		# dropout4_p=0.5, 
		hidden4_num_units=hidden_layer_size,
		hidden5_num_units=hidden_layer_size,
		output_num_units=N_EVENTS, output_nonlinearity=sigmoid,

		batch_iterator_train = BatchIterator(batch_size=1000),
		batch_iterator_test = BatchIterator(batch_size=1000),

		y_tensor_type=theano.tensor.matrix,
		update=nesterov_momentum,
		update_learning_rate=theano.shared(float(0.03)),
		update_momentum=theano.shared(float(0.9)),

		objective_loss_function=loss,
		regression=True,

		max_epochs=max_epochs,
		verbose=1,
	)

	
	###process training data####
	X = data_preprocess(X)
	total_time_points = len(X) // NO_TIME_POINTS
	no_rows = total_time_points * NO_TIME_POINTS
	
	print X.shape
	print total_time_points
	print no_rows

	X = X[0:no_rows, :]

	X = X.transpose()
	X_Samples = np.split(X, total_time_points, axis=1)
	X = np.asarray(X_Samples)
	
	print X.shape


	y = y[0:no_rows, :]
	print y.shape
	y = y[::NO_TIME_POINTS, :]
	print y.shape
	print y.dtype




	# print("Training trial %d...." %(t))
	net.fit(X,y)


	train_loss = np.array([i["train_loss"] for i in net.train_history_])
	valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
	pyplot.plot(train_loss, linewidth=2, label="train")
	pyplot.plot(valid_loss, linewidth=2, label="valid")
	pyplot.grid()
	pyplot.legend()
	pyplot.xlabel("epoch")
	pyplot.ylabel("loss")
	pyplot.ylim(0, 0.4)
	# pyplot.yscale("log")
	pyplot.show()

# 	####process test data####
# 	print("Creating prediction file ... ")

# 	X_test = X_test
# 	X_test = data_preprocess(X_test)
# 	total_test_time_points = len(X_test) // NO_TIME_POINTS
# 	remainder_test_points = len(X_test) % NO_TIME_POINTS

# 	no_rows = total_test_time_points * NO_TIME_POINTS
# 	X_test = X_test[0:no_rows, :]

# 	X_test = X_test.transpose()
# 	X_test_Samples = np.split(X_test, total_test_time_points, axis=1)
# 	X_test = np.asarray(X_test_Samples)
	
# ###########################################################################
# #######get predictions and write to files for series 9 and series 10#######
# 	print("Testing subject 0....")
# 	params = net.get_all_params_values()
# 	learned_weights = net.load_params_from(params)
# 	probabilities = net.predict_proba(X_test)

# 	total_time_points = []
# 	all_remainder_data = []
# 	subs = []
# 	total_test_points = 0

# 	trials = np.arange(1,21)

# 	for trial in trials:    	
# 		sub = 'subj{0}_series{1}'.format('0', trial)
# 		data_len = test_dict[sub]
# 		total_time_point = data_len // NO_TIME_POINTS
# 		remainder_data = data_len % NO_TIME_POINTS

# 		subs.append(sub)
# 		total_time_points.append(total_time_point)
# 		all_remainder_data.append(remainder_data)

# 	total_test_points = np.sum(total_time_points)


# 	print len(ids_tot)
# 	print cols

# 	print len(probabilities)

# 	for i, p in enumerate(probabilities):
# 		for j in range(NO_TIME_POINTS):
# 			pred_tot.append(p)


# 	print len(pred_tot)
	
# 	# for k in range(np.sum(all_remainder_data)):
# 	# 	pred_tot.append(pred_tot[-1])

# 	#submission file
# 	print('Creating submission(prediction) file...')

# 	submission_file = './test_conv_net_push.csv'
# 	# create pandas object for sbmission

# 	submission = pd.DataFrame(index=ids_tot[:len(pred_tot)],
# 	                           columns=cols,
# 	                           data=pred_tot)
# 	# write file
# 	submission.to_csv(submission_file, index_label='id', float_format='%.6f')
# 	#submission file


if __name__ == '__main__':
	build_dbn()