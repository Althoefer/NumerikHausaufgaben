# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from Blatt05_lib import system, plotxk

def cg(A, b, epsilon = 10 ** -6):
    n, _ = A.shape
    x0 = np.array([0 for i in range(n)], dtype=np.float64)
    p0 = r0 = b - A.dot(x0)

    residuals = np.array([], dtype=np.float64)
    rk = np.copy(r0)
    pk = p0
    xk = x0
    while np.linalg.norm(rk) / np.linalg.norm(r0) > epsilon:
        residuals = np.append(residuals, np.linalg.norm(rk) / np.linalg.norm(r0))
        
        alphak = np.dot(rk, rk) / np.dot(pk, A.dot(pk))
        xk1 = xk + alphak * pk
        rk1 = rk - alphak * A.dot(pk)
        
        betak = np.dot(rk1, rk1) / np.dot(rk, rk)
        pk1 = rk1 + betak * pk

        rk = rk1
        pk = pk1
        xk = xk1
    
    return xk, residuals

if __name__ == '__main__':
    print('-' * 30)
    print('CG')
    print('-' * 30)
    
    ms = [50, 100, 200]
    
    for m in ms:
        print(f'running with: m = {m} ==> n = {m ** 2}:')
    
        A, b = system(m)
        
        xk, residuals = cg(A, b)
        
        print('residuals visualized:')
        plt.plot([i for i in range(len(residuals))], residuals)
        plt.xlabel('iteration')
        plt.ylabel('residual')
        plt.show()
        plt.clf()
    
        print('solution visualized:')
        plotxk(xk)
        plt.show()
        plt.clf()
        print('-' * 30)
   