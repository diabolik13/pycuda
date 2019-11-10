import timeit

code_to_test = '''
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

a = numpy.random.randn(4,4)

a = a.astype(numpy.float32)

a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)

cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
    __global__ void doublify(float *a)
    {
      int idx = threadIdx.x + threadIdx.y*4;
      a[idx] *= 2;
    }
    """)

func = mod.get_function("doublify")
func(a_gpu, block=(4,4,1))

a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print ("original array:")
print (a)
print ("doubled with kernel:")
print (a_doubled)
'''

elapsed_time = timeit.timeit(code_to_test, number=100)/100
print(elapsed_time)

