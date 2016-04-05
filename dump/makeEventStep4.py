import numpy as np
from glob import glob
import pandas as pd
import math


def doFile():
	# trial_size = 1665
	trial_size = 1664
	trials = np.arange(1,4)
	# trials = [94]
	SUBJECT = '1'

	e_raw = []
	raw = []
	counter = 1

	raw_trial_files = glob('ICA/test/trial/'+SUBJECT+'-*.txt')
	for raw_trial_file in raw_trial_files:
		print raw_trial_file
		e = []
		dh = []

		####################### Process trial file ######################
		data = pd.read_csv(raw_trial_file,sep='\t')
		data = data.drop('NAME',axis=1)
		# print data.columns.shape
		last_col = len(data.columns)
		# print last_col
		data = data.drop(data.columns[[last_col-1]], axis=1)
		dh = list(data.columns.values)


		####################### Process event file ######################
		split1 = raw_trial_file.split('\\')
		split2 = split1[1].split('-')
		name = str(split2[0])+'-'+str(split2[1])+'-'+str(split2[2])
		raw_event_files = glob('ICA/test/eventold/'+name+'-event.txt')
	
		

		for raw_event_file in raw_event_files:
			events = pd.read_csv(raw_event_file, sep='\t', header=None)
			events = events.drop(events.columns[[0, 3]], axis=1)
			events = events.ix[1:]
			e = events.as_matrix()

		
		####################### Process new event file ######################
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

		file = open("ICA/test/event/events_"+name+".txt", "w")
		for t in table:
			# file.write(str(t[0])+"\t"+str(t[1])+"\t"+str(t[2])+"\n")
			file.write(str(t[0])+"\t"+str(t[1])+"\n")
		file.close()
		counter += 1

		


if __name__ == '__main__':
	doFile()