import numpy
import os

temp = []
path = 'data/txtshort/'
listing = os.listdir(path)
for infile in listing:
	arrays = numpy.genfromtxt(path+infile)
	temp.append(arrays)

arr = numpy.zeros((15,15,2178))


for x in range(len(temp)):
	arr[x] = temp[x]


x_train = arr[:7]
y_train = arr[:7]

x_val = arr[7:9]
y_val = arr[7:9]
x_test = arr[-1] 
y_test = arr[-1]




