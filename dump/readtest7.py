import numpy
import os

temp = []
path = 'txt/'
listing = os.listdir(path)
for infile in listing:
	arrays = numpy.genfromtxt(path+infile)
	trim = arrays[1:15]
	trim = trim[:,1:]
	temp.append(trim)
	print trim.shape


#print(len(temp))
#print(temp[0].shape)

arr = numpy.zeros((10,14,2177))

#print arr.shape

#arr[0] = temp[0]

#print arr[0]

for x in range(len(temp)):
	arr[x] = temp[x]

print arr.shape

print len(arr)




