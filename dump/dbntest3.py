import os
import time
import pandas as pd
from glob import glob

import numpy as np
from pydeeplearn.lib import deepbelief as db
from pydeeplearn.lib import restrictedBoltzmannMachine as rbm
from pydeeplearn.lib.activationfunctions import *
from pydeeplearn.lib.common import *

def load_dataset():

  # Load the files
  temp = []
  temp2 = []


  trials = np.array(['01','02','03','04','05','06','07','08','09','10'])

  for trial in trials:
    raw_event_files = glob('data/events/events_%s.txt' % (trial))
    raw_trial_files = glob('data/txtshort/0-%s-short.txt' % (trial))

    print raw_event_files
    print raw_trial_files

    for raw_event_file in raw_event_files:
      events = pd.read_csv(raw_event_file, sep='\t', header=None)
      events = events.as_matrix()
      temp2.append(events.transpose())


    for raw_trial_file in raw_trial_files:
      data = pd.read_csv(raw_trial_file,sep='\t')
      data = data.drop('NAME',axis=1)
      data = data.drop(data.columns[[2177]], axis=1)
      data = data.as_matrix()
      temp.append(data)




  X = np.zeros((10,14,2177))
  y = np.zeros((10,2,2177))

  for xr in range(len(temp)):
    X[xr] = temp[xr]
    y[xr] = temp2[xr]

  return X, y

def build_dbn():
  trainData, trainLabels = load_dataset()
  
  #testData, testLabels = load_dataset()

  # unsupervisedData = trainData

  activationFunction = Sigmoid()
  rbmActivationFunctionVisible = Sigmoid()
  rbmActivationFunctionHidden = Sigmoid()

  unsupervisedLearningRate = 0.5
  supervisedLearningRate = 0.1
  momentumMax = 0.9

  net = db.DBN(5, [2177, 100, 100, 100, 2],
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
             miniBatchSize=14,
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


  for x in range(0,len(trainData)):
    print "--start training"
    net.train(trainData[x], trainLabels[x], maxEpochs=1,
            validation=False,
            unsupervisedData=None)

  # print "complete data"
  # net.train(trainData,trainLabels,maxEpochs=1,validation=False,unsupervisedData=None)

  #probs, predicted = net.classify(testData)

  #correct = 0
  #errorCases = []



if __name__ == '__main__':
  build_dbn()