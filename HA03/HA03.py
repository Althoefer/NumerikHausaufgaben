# -*- coding: utf-8 -*-

# Exercise 4
import numpy as np

def zerlegung_ohne_pivot(A):
    A = np.copy(A)
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

def zerlegung_mit_pivot(A):
    A = np.copy(A)
    x = np.array([i for i in range(len(A))])
    for i in range(len(A)):
        # find max entry in row and exchange rows
        x[i] = i + np.argmax(abs(A[i:, i]))
        A[i, :], A[x[i], :] = A[x[i], :], np.copy(A[i, :])
        for j in range(i + 1, len(A)):
            A[j, i] = A[j, i] / A[i, i]
            A[j, i+1:] = -1 * A[j, i] * A[i, i+1:] + A[j, i+1:]
    return A, x

def permutation(p, x):
    x = np.copy(x)
    for row, row_to_swap_with in enumerate(p):
        x[row], x[row_to_swap_with] = x[row_to_swap_with], np.copy(x[row])
    return x

def vorwaerts(LU, x):
    x = np.copy(x)
    for row in range(len(LU)):
        x[row+1:] = x[row+1:] - x[row] * LU[row+1:, row]
    return x

def rueckwaerts(LU, x):
    x = np.copy(x)
    for row in reversed(range(len(LU))):
        x[row] = x[row] / LU[row, row]
        x[:row] = x[:row] - x[row] * LU[:row, row]
    return x

if __name__ == '__main__':
    print('-' * 30)
    print('LU basic tests')
    print('-' * 30)
    
    # basic tests
    A = np.array([
            [0, 0, 0, 1],
            [2, 1, 2, 0],
            [4, 4, 0, 0],
            [2, 3, 1, 0],
    ], dtype = np.float64)
    LU, p = zerlegung_mit_pivot(A)
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
        print('-' * 30)

    # advanced tests
    print('-' * 30)
    print('LU advanced tests')
    print('-' * 30)
    beta = 10
    dims = [10, 15, 20]
    for n in dims:
        # construct matrix A and rhs b
        A = np.eye(n, dtype = np.float64);
        A[0, n-1] = beta
        A[n-1, n-1] = 0.
        for i in range(n-1):
            A[i+1, i] = -1 * beta
        b = np.array([1 + beta] + [1 - beta for _ in range(n - 2)] + [-1 * beta], dtype = np.float64)
        print(f'input: {b}')
        for zerlegung in [zerlegung_ohne_pivot, zerlegung_mit_pivot]:
            LU, p = zerlegung(A)
            y = permutation(p, b)
            y = vorwaerts(LU, y)
            x = rueckwaerts(LU, y)
            print(f'solution with {zerlegung.__name__}: {x}')
        print(f'exact solution: {np.array([1 for _ in range(n)], dtype=np.float64)}')
        print('-' * 30)
