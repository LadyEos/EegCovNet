import numpy
import os

temp = []
path = 'txt/'
listing = os.listdir(path)
for infile in listing:
	arrays = numpy.genfromtxt(path+infile)
	temp.append(arrays)
#	print arrays.shape


#print(len(temp))
#print(temp[0].shape)

arr = numpy.zeros((10,15,2178))

#print arr.shape

#arr[0] = temp[0]

#print arr[0]

for x in range(len(temp)):
	arr[x] = temp[x]
#	print arr[x].shape
#	print arr[x]

print arr.shape

print type(arr)


x_train = arr[:7]
y_train = arr[:7]

x_val = arr[7:9]
y_val = arr[7:9]
x_test = arr[-1] 
y_test = arr[-1]
print x_train.shape
