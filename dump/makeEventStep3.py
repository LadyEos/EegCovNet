import numpy as np
from glob import glob
import pandas as pd
import math


def doFile():
	# trial_size = 1665
	trial_size = 1664
	trials = np.arange(1,101)
	# trials = [94]
	SUBJECT = '0'

	e_raw = []
	raw = []
	counter = 1

	for trial in trials:
		print trial
		raw_event_files = glob('ICA/event_old/'+SUBJECT+'-%s-*-event.txt' % (trial))
		raw_trial_files = glob('ICA/trial/'+SUBJECT+'-%s-*.txt' % (trial))
		
		e = []
		dh = []

		for raw_event_file in raw_event_files:
			events = pd.read_csv(raw_event_file, sep='\t', header=None)
			events = events.drop(events.columns[[0, 3]], axis=1)
			events = events.ix[1:]
			e = events.as_matrix()

		for raw_trial_file in raw_trial_files:
			print raw_trial_file
			data = pd.read_csv(raw_trial_file,sep='\t')
			data = data.drop('NAME',axis=1)
			
			# print data.columns.shape
			last_col = len(data.columns)
			# print last_col
			data = data.drop(data.columns[[last_col-1]], axis=1)
			

			dh = list(data.columns.values)

		table = np.zeros((len(dh),3))

		for x in range(len(table)):
			for y in range(len(table[x])):
				if(y == 0):
					table[x][y] = dh[x]

		length = len(e)
		counter = 0
		flag49 = False

		if(len(e) > 0):
			for t in range(len(table)):
				if(t+1 < len(table)):
					if(math.floor(table[t][0]) <= float(e[counter][0])): # If latency is not yet event latency
						if(math.floor(table[t+1][0]) > float(e[counter][0])):# if next latency is later than event latency (latency starts here)
							if(int(e[counter][1]) == 49):
								flag49 = True
							else:
								flag49 = False
							counter += 1
				table[t][1] = flag49
				if(counter >= len(e)):
					break
		else:
			print "no events"
			for t in range(len(table)):
				table[t][1] = 0.0

		file = open("ICA/event/events_"+SUBJECT+"-"+str(trial)+"-"+str(counter)+".txt", "w")
		for t in table:
			# file.write(str(t[0])+"\t"+str(t[1])+"\t"+str(t[2])+"\n")
			file.write(str(t[0])+"\t"+str(t[1])+"\n")
		file.close()
		counter += 1

		


if __name__ == '__main__':
	doFile()