# -*- coding: utf-8 -*-

# Exercise 1
import numpy as np

def zerlegung(A):
    x = np.array([i for i in range(len(A))])
    for i in range(len(A)):
        # exchange rows if 0 on main diagonal
        if A[i, i] == 0:
            for k in range(i + 1, len(A)):
                if A[k, i] != 0:
                    x[i] = k
                    A[i, :], A[k, :] = A[k, :], np.copy(A[i, :])
                    break
        for j in range(i + 1, len(A)):
            A[j, i] = A[j, i] / A[i, i]
            A[j, i+1:] = -1 * A[j, i] * A[i, i+1:] + A[j, i+1:]
    return A, x

def permutation(p, x):
    for row, row_to_swap_with in enumerate(p):
        x[row], x[row_to_swap_with] = x[row_to_swap_with], np.copy(x[row])
    return x

def vorwaerts(LU, x):
    for row in range(len(LU)):
        x[row+1:] = x[row+1:] - x[row] * LU[row+1:, row]
    return x

def rueckwaerts(LU, x):
    for row in reversed(range(len(LU))):
        x[row] = x[row] / LU[row, row]
        x[:row] = x[:row] - x[row] * LU[:row, row]
    return x

if __name__ == '__main__':
    print('-' * 30)
    print('LU')
    print('-' * 30)
    
    # basic tests
    A = np.array([
            [0, 0, 0, 1],
            [2, 1, 2, 0],
            [4, 4, 0, 0],
            [2, 3, 1, 0],
    ], dtype = np.float64)
    LU, p = zerlegung(A)
    rhs = [
        np.array([3, 5, 4, 5], dtype = np.float64),
        np.array([4, 10, 12, 11], dtype = np.float64),
    ]
    for b in rhs:
        print(f'input: {b}')
        b = permutation(p, b)
        b = vorwaerts(LU, b)
        x = rueckwaerts(LU, b)
        print(f'solution: {x}')

    # accuracy tests
    for n in [5, 10, 15, 20]:
        A = np.array([[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, n + 1)], dtype = np.float64)
        b = np.array([1 / (i + 1) for i in range(1, n + 1)], dtype = np.float64)
        LU, p = zerlegung(A)
        print(f'input: {b}')
        b = permutation(p, b)
        b = vorwaerts(LU, b)
        x = rueckwaerts(LU, b)
        print(f'solution: {x}')
    print('-' * 30)

# Exercise 2
def sherman_morris(LU, p, u, v, b_dach):
    # calculate z from LGS: A * z = u
    u = permutation(p, u)
    u = vorwaerts(LU, u)
    z = rueckwaerts(LU, u)
    
    # calculate alpha
    try:
        alpha = 1 / (1 + np.dot(v, z))
    except ZeroDivisionError:
        print('Matrix A_dach is not regular')
        return None
    
    # calculate z_dach from LGS: A * z_dach = b_dach
    b_dach = permutation(p, b_dach)
    b_dach = vorwaerts(LU, b_dach)
    z_dach = rueckwaerts(LU, b_dach)
    
    # calculate final solution x_dach
    x_dach = z_dach - alpha * np.dot(v, z_dach) * z
    return x_dach

if __name__ == '__main__':
    print('-' * 30)
    print('Sherman-Morris')
    print('-' * 30)

    A = np.array([
            [0, 0, 0, 1],
            [2, 1, 2, 0],
            [4, 4, 0, 0],
            [2, 3, 1, 0],
    ], dtype = np.float64)
    u = np.array([0, 1, 2, 3], dtype = np.float64)
    v = np.array([0, 0, 0, 1], dtype = np.float64)
    b_dach = np.array([3, 5, 4, 5], dtype = np.float64)

    # we already have LU (and p) for A
    LU, p = zerlegung(A)

    print(f'input: u: {u} v: {v}')
    x_dach = sherman_morris(LU, p, u, v, b_dach)
    print(f'solution: {x_dach}')
    print('-' * 30)

# Exercise 3
from math import sqrt

def cholesky(A):
    ncols = len(A[0, :])
    for i in range(ncols):
        A[0, i] = sqrt(A[0, i])
        A[1, i] = A[1, i] / A[0, i]
        if i + 1 < ncols:
            A[0, i + 1] = A[0, i + 1] - A[1, i] ** 2
    return A

def vorwaerts(L, x):
    ncols = len(A[0, :])
    for i in range(ncols):
        x[i] = x[i] / A[0, i]
        if i + 1 < ncols:
            x[i + 1] = x[i + 1] - x[i] * A[1, i]
    return x

def rueckwaerts(L, x):
    for i in reversed(range(len(A[0, :]))):
        x[i] = x[i] / A[0, i]
        if i - 1 >= 0:
            x[i - 1] = x[i - 1] - x[i] * A[1, i - 1]
    return x

if __name__ == '__main__':
    print('-' * 30)
    print('Cholesky')
    print('-' * 30)
    for n in [4, 100, 1000, 10000]:
        # last -1 added for symmetry
        A = np.array([
                [2.] * n, # main diagonal
                [-1.] * n, # one of adjacent symmetric diagonals
        ], dtype = np.float64)
        L = cholesky(A)
        b = np.array([-1 / (n + 1) ** 2 for _ in range(n)], dtype = np.float64)
        print(f'input: {b}')
        b = vorwaerts(L, b)
        x = rueckwaerts(L, b)
        print(f'solution: {x}')
    print('-' * 30)
