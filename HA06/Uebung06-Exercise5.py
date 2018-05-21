# -*- coding: utf-8 -*-
import math
import numpy as np

def newton_multi_dim(f, f_deriv, x0, value, iterations=20):
    x = x0
    for _ in range(iterations):
        x = x - np.dot(np.linalg.inv(f_deriv(x)), f(x) - value)
    return x

def f(vec):
    x, y = vec
    return math.sin(x) - y, math.e ** (-1 * y) - x

def f_deriv(vec):
    # Jacobi matrix, which fits to functions of f
    J = np.array([
        [lambda x, y: math.cos(x), lambda x, y: -1],
        [lambda x, y: -1, lambda x, y: -1 * math.e ** (-1 * y)],
    ])
    nrows, ncols = J.shape

    ret = np.zeros(shape=(nrows, ncols), dtype=np.float64)
    for row in range(nrows):
        for col in range(ncols):
            ret[row, col] = J[row, col](*vec)
    return ret


if __name__ == '__main__':
    print('-' * 30)
    print('Newton Verfahren mit mehrdimensionalem Vektor')
    print('-' * 30)

    iterations = [i for i in range(0, 10)]

    columns = ['#Iterations', 'Newton-Approximation']
    columns_length = [len(s) for s in columns]
    print(' '.join(columns))
    
    dim = 2
    x0 = np.array([0 for _ in range(dim)], dtype=np.float64)
    value = np.array([0 for _ in range(dim)], dtype=np.float64)
    
    np.set_printoptions(precision=15)
    for iteration in iterations:
        x_newton = newton_multi_dim(f, f_deriv, x0, value, iterations=iteration)
        print(f'{iteration:{columns_length[0]}d} {x_newton}')

    print('-' * 30)
