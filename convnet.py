import cPickle as pickle
from datetime import datetime
import numpy as np
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import theano
from theano.tensor.nnet import sigmoid
from sklearn.preprocessing import StandardScaler
from lasagne.objectives import aggregate, binary_crossentropy
from matplotlib import pyplot

def float32(k):
	return np.cast['float32'](k)

class AdjustVariable(object):
	def __init__(self, name, start=0.03, stop=0.001):
		self.name = name
		self.start, self.stop = start, stop
		self.ls = None
	def __call__(self, nn, train_history):
		if self.ls is None:
			self.ls = np.linspace(
				self.start, self.stop,nn.max_epochs)
		
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
			print("Best valid loss was {:.6f} at epoch {}.".format(self.best_valid, self.best_valid_epoch))
		
		nn.load_params_from(self.best_weights)
		raise StopIteration()

def loss(x, t):
	return aggregate(binary_crossentropy(x, t))

def plot(net):
	train_loss = np.array([i["train_loss"] for i in net.train_history_])
	valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
	pyplot.plot(train_loss, linewidth=3, label="train")
	pyplot.plot(valid_loss, linewidth=3, label="valid")
	pyplot.grid()
	pyplot.legend()
	pyplot.xlabel("epoch")
	pyplot.ylabel("loss")
	pyplot.show()

def build_dbn():
	cols = ['Start']

	t = 1
	channels = 14
	batch_size = None #None = arbitary batch size
	hidden_layer_size = 100
	N_EVENTS = 1
	max_epochs = 500
	NO_TIME_POINTS = 177
	
	ids_tot = []
	pred_tot = []
	test_dict = dict()
	
	test_total = 0
	
	net = NeuralNet(
		layers=[
			('input', layers.InputLayer),
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
		input_shape=(batch_size, channels, NO_TIME_POINTS),
		conv1_num_filters=4, conv1_filter_size=1,
		conv1_nonlinearity=None,
		conv2_num_filters=8, conv2_filter_size=5,
		pool1_pool_size=4,
		dropout2_p=0.5, hidden4_num_units=hidden_layer_size,
		dropout3_p=0.3, hidden5_num_units=hidden_layer_size,
		dropout4_p=0.2, output_num_units=N_EVENTS,
		output_nonlinearity=sigmoid,
		
		batch_iterator_train = BatchIterator(batch_size=1000),
		batch_iterator_test = BatchIterator(batch_size=1000),
		
		y_tensor_type=theano.tensor.matrix,
		update=nesterov_momentum,
		update_learning_rate=theano.shared(float(0.03)),
		update_momentum=theano.shared(float(0.9)),
		
		objective_loss_function=loss,
		regression=True,

		on_epoch_finished=[
			AdjustVariable('update_learning_rate', start=0.03,stop=0.0001),
			AdjustVariable('update_momentum', start=0.9, stop=0.999),
			EarlyStopping(patience=100),	
		],

		max_epochs=max_epochs,
		verbose=1,
		)
	
	# load trial dataset
	dic = pickle.load(open('datapickled/traildata.pickle', 'rb'))
	
	X = dic['X']
	y = dic['y']
	
	# process training data
	total_time_points = len(X) // NO_TIME_POINTS
	no_rows = total_time_points * NO_TIME_POINTS

	X = X[0:no_rows, :]
	
	X = X.transpose()
	X_Samples = np.split(X, total_time_points, axis=1)
	X = np.asarray(X_Samples)
	
	y = y[0:no_rows, :]
	y = y[::NO_TIME_POINTS, :]
	y = y.astype('float32')
	
	net.fit(X,y)
	
	tip = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	
	# Save the net
	with open('net/net'+tip+'.pickle', 'wb') as f:
		pickle.dump(net, f, -1)
	
	plot(net)

	# Load test data
	dic = pickle.load(open('datapickled/testdata2.pickle', 'rb'))
	X_test = dic['X_test']
	ids_tot = dic['ids_tot']
	test_dict = dic['test_dict']
	test_total = dic['test_total']

	####process test data####
	print("Creating prediction file ... ")
	
	X_test = X_test
	total_test_len = len(X_test)
	
	total_test_time_points = len(X_test) // NO_TIME_POINTS
	remainder_test_points = len(X_test) % NO_TIME_POINTS
	
	no_rows = total_test_time_points * NO_TIME_POINTS
	X_test = X_test[0:no_rows, :]

	X_test = X_test.transpose()
	X_test_Samples = np.split(X_test, total_test_time_points, axis=1)
	X_test = np.asarray(X_test_Samples)
	
	# Evaluate test data
	print("Testing subject 0....")
	params = net.get_all_params_values()
	learned_weights = net.load_params_from(params)
	probabilities = net.predict_proba(X_test)
	
	total_test_points = total_test_len // NO_TIME_POINTS
	remainder_data = total_test_len % NO_TIME_POINTS
	for i, p in enumerate(probabilities):
		if i != total_test_points:
			for j in range(NO_TIME_POINTS):
				pred_tot.append(p)
	
	# create prediction file
	print('Creating submission(prediction) file...')
	tip = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	submission_file = 'res/test_conv_net_push'+tip+'.csv'
	# create pandas object
	submission =  pd.DataFrame(index=ids_tot[:len(pred_tot)],columns=cols,data=pred_tot)
	# write file
	submission.to_csv(submission_file, index_label='id', float_format='%.6f')

if __name__ == '__main__':
	build_dbn()