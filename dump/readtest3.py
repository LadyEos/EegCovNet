import numpy
import os

temp = []
path = 'txt/'
listing = os.listdir(path)
for infile in listing:
	arrays = [numpy.array(map(None, line.split())) for line in open(path+infile)]
	temp2 = numpy.array(arrays)
	temp.append(temp2)


x = numpy.asarray(temp)
print(x.shape)

print(x[0][0].shape)
print(x[0][1].shape)