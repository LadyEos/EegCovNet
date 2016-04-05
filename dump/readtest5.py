import numpy
import os

temp = []
path = 'txt/'
listing = os.listdir(path)
for infile in listing:
	arrays = numpy.genfromtxt(path+infile)
	temp.append(numpy.array([arrays]))


print(len(temp))


a = numpy.ones((15,1))
b = numpy.ones((15,3585))
c = numpy.ones((1,1))

