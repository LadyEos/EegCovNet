import numpy

fileName = "txt/0-01.txt"

data = numpy.genfromtxt(fileName) 
#arrays = [numpy.array(map(None, line.split())) for line in open(fileName)]
nolabel = data[:,1:]

print(nolabel)
print(nolabel.shape)
#print(arrays)