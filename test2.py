import timeit

code_to_test = '''

import numpy

a = numpy.random.randn(4,4)

a = a.astype(numpy.float32)

a_doubled = numpy.empty_like(a)

a_doubled = a * 2

print ("original array:")
print (a)
print ("doubled with kernel:")
print (a_doubled)
'''

elapsed_time = timeit.timeit(code_to_test, number=100)/100
print(elapsed_time)