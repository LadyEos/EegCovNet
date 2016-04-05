import numpy as np
from glob import glob
import pandas as pd
import math


def doFile():
	trial_size = 1665
	# trials = np.arange(1,80)
	trials = [1,2]
	e_raw = []
	raw = []

	for trial in trials:
		raw_event_files = glob('data2/0-%s-event.txt' % (trial))
		raw_trial_files = glob('data2/0-%s.txt' % (trial))
		
		e = []
		dh = []

		for raw_event_file in raw_event_files:
			events = pd.read_csv(raw_event_file, sep='\t', header=None)
			events = events.drop(events.columns[[0, 3]], axis=1)
			events = events.ix[1:]
			e = events.as_matrix()

		print e

		for raw_trial_file in raw_trial_files:
			data = pd.read_csv(raw_trial_file,sep='\t')
			data = data.drop('NAME',axis=1)
			data = data.drop(data.columns[[trial_size]], axis=1)
			dh = list(data.columns.values)

		table = np.zeros((len(dh),3))

		for x in range(len(table)):
			for y in range(len(table[x])):
				if(y == 0):
					table[x][y] = dh[x]

		length = len(e)
		counter = 0
		flag49 = False
		flag50 = False
		flagLatency = True

		if(len(e) > 0):
			for t in range(len(table)):
				if(flagLatency):
					if(t+1 < len(table)):
						if(math.floor(table[t][0]) <= float(e[counter][0])): # If latency is not yet event latency
							if(math.floor(table[t+1][0]) > float(e[counter][0])):# if next latency is later than event latency (latency starts here)
								if(int(e[counter][1]) == 49):
									flag49 = True
									flag50 = False
								elif(int(e[counter][1]) == 50):
									flag49 = False
									flag50 = True
								counter += 1
				print flag49, flag50
				table[t][1] = flag49
				table[t][2] = flag50
				if(counter >= len(e)):
					flagLatency = False
			
			file = open("events_"+str(trial)+".txt", "w")
			for t in table:
				file.write(str(t[0])+"\t"+str(t[1])+"\t"+str(t[2])+"\n")
			file.close()
		else:
			print "no events"


		


if __name__ == '__main__':
	doFile()