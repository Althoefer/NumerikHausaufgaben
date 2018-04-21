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
    # basic tests
    A = np.array([
            [0, 0, 0, 1],
            [2, 1, 2, 0],
            [4, 4, 0, 0],
            [2, 3, 1, 0],
    ])
    LU, p = zerlegung(A)
    rhs = [
        np.array([3, 5, 4, 5]),
        np.array([4, 10, 12, 11]),
    ]
    for b in rhs:
        print(f'input: {b}')
        b = permutation(p, b)
        b = vorwaerts(LU, b)
        x = rueckwaerts(LU, b)
        print(f'solution: {x}')

    # accuracy tests
    for n in [5, 10, 15, 20]:
        A = np.array([[1 / (i + j - 1) for j in range(1, n + 1)] for i in range(1, n + 1)])
        b = np.array([1 / (i + 1) for i in range(1, n + 1)])
        LU, p = zerlegung(A)
        print(f'input: {b}')
        b = permutation(p, b)
        b = vorwaerts(LU, b)
        x = rueckwaerts(LU, b)
        print(f'solution: {x}')
