from datetime import datetime
import numpy as np
import pandas as pd
from glob import glob
import math

FILENAME = 'res/test_conv_net_push2015-10-13-11-52-52.csv'
raw = glob(FILENAME)
probs = pd.read_csv(raw[0], sep=',', header=0)
probs= probs.as_matrix()
probs = [[n.split('_'),p] for n,p in probs]

dataframes = [[n[0],n[1],n[2], "{0:.3f}".format(float(n[2])/1000) if
n[2] !='0.0000' else n[2], p] for n, p in probs]
ones = [[sub,t,f,sec, int(math.floor(float(sec))) ,p] for sub,t,f,sec,p in
dataframes if p >= 0.055]

testss = set(e[1] for e in ones)
tests = list(testss)
tests = sorted(tests)
results =[]

for t in tests:
	tset = set(x[4] for x in ones if x[1] == t )
	temp = [t,sorted(list(tset))]
	results.append(temp)
for r in results:
	print r