# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:51:34 2019

@author: Shahir
"""

import math
import cmath

import numpy as np

def dft_basic(x):
    N = len(x)
    return [sum([x[n]*cmath.exp(-2j*math.pi*k*n/N) for n in range(N)])/N for k in range(N)]

def dft_symmetric(x):
    N = len(x)
    N2 = N//2 + (N % 2) + 1
    dft1 = [sum([x[n]*cmath.exp(-2j*math.pi*k*n/N) for n in range(N)])/N for k in range(N2)]
    dft2 = [n.conjugate() for n in dft1[1:N//2][::-1]]
    return dft1 + dft2

def dft_recursive(x):
    N = len(x)
    if N == 1:
        return x
    X_even = dft_recursive(x[::2])
    X_odd = dft_recursive(x[1::2])
    return [(X_even[k % (N//2)] + X_odd[k % (N//2)]*cmath.exp(-2j*math.pi*k/N))/2 for k in range(N)]

def idft_basic(X):
    N = len(X)
    return [sum([X[k]*math.e**(2j*math.pi*k*n/N) for k in range(N)]) for n in range(N)]

def idft_recursive(X):
    N = len(X)
    if N == 1:
        return X
    x_even = idft_recursive(X[::2])
    x_odd = idft_recursive(X[1::2])
    return [x_even[k % (N//2)] + x_odd[k % (N//2)]*cmath.exp(2j*math.pi*k/N) for k in range(N)]

if __name__ == '__main__':
    import random
    import time
    
    x = [random.random() for n in range(2**10)]
    start_time = time.time()
    X1 = dft_basic(x)
    print("dft_basic took", time.time() - start_time, "seconds")
    
    start_time = time.time()
    X2 = dft_symmetric(x)
    print("dft_symmetric took", time.time() - start_time, "seconds")
    if not np.allclose(X1, X2):
        print("dft_symmetric failed!")
    
    start_time = time.time()
    X3 = dft_recursive(x)
    print("dft_recursive took", time.time() - start_time, "seconds")
    if not np.allclose(X1, X3):
        print("dft_recursive failed!")
    
    start_time = time.time()
    x1 = idft_basic(X1)
    print("idft_basic took", time.time() - start_time, "seconds")
    if not np.allclose(x, x1):
        print("idft_basic failed!")
    
    start_time = time.time()
    x2 = idft_recursive(X1)
    print("idft_recusrive took", time.time() - start_time, "seconds")
    if not np.allclose(x, x2):
        print("idft_recusrive failed!")