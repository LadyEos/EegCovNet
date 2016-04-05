import numpy
import os

temp = []
path = 'txt/'
listing = os.listdir(path)
for infile in listing:
	arrays = numpy.genfromtxt(path+infile)
	temp.append(arrays)
print(len(temp))
data = numpy.array(temp)
print(data.shape)