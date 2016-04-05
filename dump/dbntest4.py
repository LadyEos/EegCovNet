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
from pydeeplearn.lib import deepbelief as db
from pydeeplearn.lib import restrictedBoltzmannMachine as rbm
from pydeeplearn.lib.activationfunctions import *
from pydeeplearn.lib.common import *
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
  NO_TIME_POINTS = 100


  ids_tot = []
  pred_tot = []
  test_dict = dict()

  test_total = 0


  X, y = load_dataset(SIZE)
  X_test, ids_tot, test_dict, test_total = load_testdata(SIZE)

  activationFunction = Sigmoid()
  rbmActivationFunctionVisible = Sigmoid()
  rbmActivationFunctionHidden = Sigmoid()

  unsupervisedLearningRate = 0.5
  supervisedLearningRate = 0.1
  momentumMax = 0.9

  net = db.DBN(5, [1665, 100, 100, 100, 2],
             binary=False,
             activationFunction=activationFunction,
             rbmActivationFunctionVisible=rbmActivationFunctionVisible,
             rbmActivationFunctionHidden=rbmActivationFunctionHidden,
             unsupervisedLearningRate=unsupervisedLearningRate,
             supervisedLearningRate=supervisedLearningRate,
             momentumMax=momentumMax,
             nesterovMomentum=True,
             rbmNesterovMomentum=True,
             rmsprop=False,
             miniBatchSize=1000,
             hiddenDropout=0,
             visibleDropout=0,
             momentumFactorForLearningRateRBM=False,
             firstRBMheuristic=False,
             rbmVisibleDropout=0,
             rbmHiddenDropout=0,
             preTrainEpochs=2,
             sparsityConstraintRbm=False,
             sparsityRegularizationRbm=0.001,
             sparsityTragetRbm=0.01)


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
  net.train(X, y, maxEpochs=1, validation=False, unsupervisedData=None)


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



if __name__ == '__main__':
  build_dbn()