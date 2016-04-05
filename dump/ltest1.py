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

def load_dataset():

	# Load the files
	raw = []
	y_raw = []


	trials = np.array(['01','02','03','04','05','06','07','08','09','10'])

	for trial in trials:
		raw_event_files = glob('data/events/events_%s.txt' % (trial))
		raw_trial_files = glob('data/txtshort/0-%s-short.txt' % (trial))

		for raw_event_file in raw_event_files:
			events = pd.read_csv(raw_event_file, sep='\t', header=None)
			#events = events.ix[1:]
			events = events.drop(events.columns[[0]],axis=1)
			events = events.as_matrix()
			y_raw.append(events)

		for raw_trial_file in raw_trial_files:
			data = pd.read_csv(raw_trial_file,sep='\t')
			data = data.drop('NAME',axis=1)
			data = data.drop(data.columns[[2177]], axis=1)
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

def build_dbn():
	X, y = load_dataset()
	t = 1
	channels = 14
	batch_size = None  #None = arbitary batch size
	hidden_layer_size = 100  #change to 1024
	N_EVENTS = 1
	max_epochs = 100
	NO_TIME_POINTS = 100

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

		max_epochs=max_epochs,
		verbose=1,
	)

	
		
	total_time_points = len(X) // NO_TIME_POINTS

	no_rows = total_time_points * NO_TIME_POINTS
	X = X[0:no_rows, :]

	print X.shape

	X = X.transpose()
	X_Samples = np.split(X, total_time_points, axis=1)
	X = np.asarray(X_Samples)
	print X.shape


	y = y[0:no_rows, :]
	y = y[::NO_TIME_POINTS, :]





	print("Training trial %d...." %(t))
	net.fit(X,y)
	

	# print "complete data"
	# net.train(trainData,trainLabels,maxEpochs=1,validation=False,unsupervisedData=None)

	#probs, predicted = net.classify(testData)

	#correct = 0
	#errorCases = []



if __name__ == '__main__':
	build_dbn()