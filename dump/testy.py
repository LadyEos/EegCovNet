import numpy as np

y = np.arange(333000).reshape((166500,2))
print y.shape
print y

NO_TIME_POINTS = 100
total_time_points = len(y) // NO_TIME_POINTS
no_rows = total_time_points * NO_TIME_POINTS

print total_time_points
print no_rows

y2 = y[0:no_rows, :]

print y2.shape
print y

y3 = y2[::NO_TIME_POINTS, :]

print y3.shape
print y3
