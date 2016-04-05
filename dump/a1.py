import numpy as np

a = [[-0.0785, -0.0809, -0.0834,  0.0313,  0.0304,  0.0316],
	[-0.0767, -0.0743, -0.0704, -0.0136, -0.0118, -0.0101]]

data = np.asarray(a)

mins = data.min(axis=1)
maxs = data.max(axis=1)

print "min"
print(mins)
print "max"
print(maxs)