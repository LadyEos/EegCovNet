import numpy as np
import pandas as pd
from glob import glob
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import theano
from theano.tensor.nnet import sigmoid
from sklearn.preprocessing import StandardScaler
from lasagne.objectives import aggregate, binary_crossentropy
from matplotlib import pyplot
try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer

def float32(k):
	return np.cast['float32'](k)

class AdjustVariable(object):
	def __init__(self, name, start=0.03, stop=0.001):
		self.name = name
		self.start, self.stop = start, stop
		self.ls = None

	def __call__(self, nn, train_history):
		if self.ls is None:
			self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

		epoch = train_history[-1]['epoch']
		new_value = np.cast['float32'](self.ls[epoch - 1])
		getattr(nn, self.name).set_value(new_value)

class EarlyStopping(object):
	def __init__(self, patience=100):
		self.patience = patience
		self.best_valid = np.inf
		self.best_valid_epoch = 0
		self.best_weights = None

	def __call__(self, nn, train_history):
		current_valid = train_history[-1]['valid_loss']
		current_epoch = train_history[-1]['epoch']
		if current_valid < self.best_valid:
			self.best_valid = current_valid
			self.best_valid_epoch = current_epoch
			self.best_weights = nn.get_all_params_values()
		elif self.best_valid_epoch + self.patience < current_epoch:
			print("Early stopping.")
			print("Best valid loss was {:.6f} at epoch {}.".format(
				self.best_valid, self.best_valid_epoch))
			nn.load_params_from(self.best_weights)
			raise StopIteration()

def load_dataset(SIZE, raw_trial_files,raw_event_files):

	# Load the files
	raw = []
	y_raw = []
	
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
	N_EVENTS = 1
	max_epochs = 500
	NO_TIME_POINTS = 100


	ids_tot = []
	pred_tot = []
	test_dict = dict()

	test_total = 0
	
	net = NeuralNet(
		layers=[
			('input', layers.InputLayer),
			('dropout1', layers.DropoutLayer),
			('conv1', layers.Conv1DLayer),
			('conv2', layers.Conv1DLayer),
			('pool1', layers.MaxPool1DLayer),
			('dropout2', layers.DropoutLayer),
			('hidden4', layers.DenseLayer),
			('dropout3', layers.DropoutLayer),
			('hidden5', layers.DenseLayer),
			('dropout4', layers.DropoutLayer),
			('output', layers.DenseLayer),
		],
		input_shape=(None, channels, NO_TIME_POINTS),
		dropout1_p=0.5,
		conv1_num_filters=4, conv1_filter_size=1,
		conv2_num_filters=8, conv2_filter_size=4, pool1_pool_size=4,
		dropout2_p=0.5, hidden4_num_units=hidden_layer_size,
		dropout3_p=0.5, hidden5_num_units=hidden_layer_size,
		dropout4_p=0.5, output_num_units=N_EVENTS, output_nonlinearity=sigmoid,

		batch_iterator_train = BatchIterator(batch_size=1000),
		batch_iterator_test = BatchIterator(batch_size=1000),

		y_tensor_type=theano.tensor.matrix,
		update=nesterov_momentum,
		update_learning_rate=theano.shared(float(0.03)),
		update_momentum=theano.shared(float(0.9)),

		objective_loss_function=loss,
		regression=True,

		on_epoch_finished=[
	        # AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
	        # AdjustVariable('update_momentum', start=0.9, stop=0.999),
	        EarlyStopping(patience=50),
	        ],

		max_epochs=max_epochs,
		verbose=1,
	)

	trials = np.arange(1,101)

	

	for trial in trials:

		raw_trial_files = glob('data2/trial/0-%s.txt' % (trial))
		raw_event_files = glob('data2/event/events_0-%s.txt' % (trial))

		X, y = load_dataset(SIZE,raw_trial_files,raw_event_files)
		# X_test, ids_tot, test_dict, test_total = load_testdata(SIZE)
		
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

		y = y.astype('float32')
		print y.dtype



		# print("Training trial %d...." %(t))
		net.fit(X,y)


	train_loss = np.array([i["train_loss"] for i in net.train_history_])
	valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
	pyplot.plot(train_loss, linewidth=3, label="train")
	pyplot.plot(valid_loss, linewidth=3, label="valid")
	pyplot.grid()
	pyplot.legend()
	pyplot.xlabel("epoch")
	pyplot.ylabel("loss")
	pyplot.ylim(0, 1)
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