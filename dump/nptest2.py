import numpy

x = numpy.array([['a','b','c'],['d','e','f'],['g','h','i'],['j','k','l'],['m','n','o'],['p','q','r'],['s','t','u'],['v','w','x'],['y','z','1'],['2','3','4'],['5','6','7'],['8','9','10']])
y = numpy.array([['a1','b1','c1'],['d1','e1','f1'],['g1','h1','i1'],['j1','k1','l1'],['m1','n1','o1'],['p1','q1','r1'],['s1','t1','u1'],['v1','w1','x1'],['y1','z1','11'],['21','31','41']])
z = numpy.array([['a2','b2','c2'],['d2','e2','f2'],['g2','h2','i2'],['j2','k2','l2'],['m2','n2','o2'],['p2','q2','r2'],['s2','t2','u2'],['v2','w2','x2'],['y2','z2','12']])

lst = [x,y,z]

data = numpy.asarray(lst)
print(data.shape)
print(data)