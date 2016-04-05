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

dataframes = [[n[0],n[1],n[2], "{0:.3f}".format(float(n[2])/1000) if n[2] !='0.0000' else n[2], p] for n, p in probs]

ones = [[sub,t,f,sec, int(math.floor(float(sec))) ,p] for sub,t,f,sec,p in dataframes if p  >= 0.055]

# print ones[0]


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




# for x in range(len(ones)):
# 	if(y[x][1] == 'test5'):
# 		# print y[x][1], "{0:.3f}".format(float(y[x][3])), y[x][4]
# 		print y[x][1], y[x][4], y[x][5]


# tip = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# submission_file = 'probs/results'+tip+'.csv'
# # create pandas object for sbmission

# submission = pd.DataFrame(results)
# # write file
# submission.to_csv(submission_file, float_format='%.3f')
# #submission file