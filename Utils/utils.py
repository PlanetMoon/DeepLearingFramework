# encoding: utf-8

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit # very important, not import will lead to an initial error
from pycuda.compiler import SourceModule


def dot(x, y):
    return np.dot(x, y)


def sqrt(x):
    return np.sqrt(x)


def dot_GPU(x, y):
    (xh, xw) = x.shape
    (yh, yw) = y.shape
    xw = np.int32(xw)
    yw = np.int32(yw)
    x.astype(np.float32)
    y.astype(np.float32)

    mod = SourceModule("""
    __global__ void mat_mut(float *a, float *b, float *c, unsigned int p, unsigned int q)
    {
      int x = threadIdx.x + blockDim.x*blockIdx.x;
      int y = threadIdx.y + blockDim.y*blockIdx.y;
      c[y + x*q] = 0;
      for(int i = 0; i < p; i++)
      {
        c[y + x*q] = c[y + x*q] + a[i + x*p]*b[y + i*q];
      }
    }
    """)

    c = np.zeros((xh, yw)).astype(np.float32)
    func = mod.get_function("mat_mut")
    func(cuda.InOut(x), cuda.InOut(y), cuda.InOut(c), xw, yw, block=(xh, 1, 1), grid=(1, 1))
    return c
