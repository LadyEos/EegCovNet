import os
import time

import numpy as np
from pydeeplearn.lib import deepbelief as db
from pydeeplearn.lib import restrictedBoltzmannMachine as rbm
from pydeeplearn.lib.activationfunctions import *
from pydeeplearn.lib.common import *

def load_dataset():

  # Load the files
  temp = []
  temp2 = []
  path = 'data/txtshort/'
  listing = os.listdir(path)
  for infile in listing:
    arrays = np.genfromtxt(path+infile)
    trim = arrays[:,1:]
    trim = trim[1:15]
    temp.append(trim)
    for x in range(len(trim)):
      temp2.append(x)

  arr = np.zeros((10,14,2177))
  targ = np.zeros((14,2177))
  new = np.zeros((140,2177))

  for xr in range(len(temp)):
    if(xr == 4):
      targ = temp[xr]
    arr[xr] = temp[xr]

    for xx in range(len(temp2)):
      new[xx]= temp2[xx]

  # flat = np.zeros((10,30478))
  # for x in range(len(arr)):
  #   flat[x] = arr[x].flatten()

  #y = [1,2,3,4,5,6,7,8,9,10]
  #y = [[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]]
  y = np.ones_like(arr)
  yy = np.ones_like(new)
  #y = [[0],[1],[0],[0],[1],[0],[0],[1],[1],[0]]
  for x in range(len(y)):
    y[x] = targ

  d = 0
  while d < len(yy):
    for c in range(len(targ)):
      yy[d]=targ[c]
      d += 1

  print new.shape
  print yy.shape

  return new, yy

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

  net = db.DBN(5, [2177, 100, 100, 100, 2177],
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


  # for x in range(0,len(trainData)):
  #   print "--start training"
  #   net.train(trainData[x], trainLabels[x], maxEpochs=1,
  #           validation=False,
  #           unsupervisedData=None)

  print "complete data"
  net.train(trainData,trainLabels,maxEpochs=100,validation=False,unsupervisedData=None)

  #probs, predicted = net.classify(testData)

  #correct = 0
  #errorCases = []



if __name__ == '__main__':
  build_dbn()