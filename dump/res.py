import cPickle as pickle
from datetime import datetime
from time import gmtime, strftime
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
	pyplot.ylim(0, 1)
	# pyplot.yscale("log")
	pyplot.show()

def data_preprocess(X):
    # normalize data
    mean = X.mean(axis=0)
    X -= mean
    std = X.std(axis=0)
    X /= std
    X_prep = X
    return X_prep

def load_testdata(SIZE):
	print("Reading test data .... ")
	test_dict = dict()
	test_total = 0
	ids_tot = []
	test = []
	idt = []
	idx = []

	trials = np.arange(1,21)


	raw_trial_files = glob('data2/ICA/test/1-*.txt')

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


#################################################################################################
SIZE = 1665
NO_TIME_POINTS = 177
ids_tot = []
pred_tot = []
test_dict = dict()

cols = ['Start']

test_total = 0

# net = pickle.load(open('net/net2015-10-09-16-14-08.pickle', 'rb'))
net = pickle.load(open('net/net2015-10-12-21-53-18.pickle', 'rb'))
# plot(net)

# X_test, ids_tot, test_dict, test_total = load_testdata(SIZE)
dic = pickle.load(open('datapickled/testdata2.pickle', 'rb'))
X_test = dic['X_test']
ids_tot = dic['ids_tot']
test_dict = dic['test_dict']
test_total = dic['test_total']

####process test data####
print("Creating prediction file ... ")

X_test = X_test
# X_test = data_preprocess(X_test)

print X_test.shape

total_test_len = len(X_test)

total_test_time_points = len(X_test) // NO_TIME_POINTS
remainder_test_points = len(X_test) % NO_TIME_POINTS

no_rows = total_test_time_points * NO_TIME_POINTS
X_test = X_test[0:no_rows, :]

X_test = X_test.transpose()
X_test_Samples = np.split(X_test, total_test_time_points, axis=1)
X_test = np.asarray(X_test_Samples)

print X_test.shape


###########################################################################
#######get predictions and write to files for series 9 and series 10#######
print("Testing subject 0....")
params = net.get_all_params_values()
learned_weights = net.load_params_from(params)
probabilities = net.predict_proba(X_test)

print probabilities
print probabilities.shape


total_test_points = total_test_len // NO_TIME_POINTS
remainder_data = total_test_len % NO_TIME_POINTS




for i, p in enumerate(probabilities):
	if i != total_test_points:
		for j in range(NO_TIME_POINTS):
			pred_tot.append(p)

#submission file
print('Creating submission(prediction) file...')
tip = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
submission_file = 'res/test_conv_net_push'+tip+'.csv'
# create pandas object for sbmission

submission = pd.DataFrame(index=ids_tot[:len(pred_tot)],columns=cols,data=pred_tot)
# write file
submission.to_csv(submission_file, index_label='id', float_format='%.6f')
#submission file


