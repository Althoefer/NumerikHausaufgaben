# -*- coding: utf-8 -*-

# Exercise 1
import numpy as np


def zerlegung_mit_pivot(A):
    A = np.copy(A)
    x = np.array([i for i in range(len(A))])
    for i in range(len(A)):
        # find max entry in row and exchange rows
        x[i] = i + np.argmax(abs(A[i:, i]))
        A[i, :], A[x[i], :] = A[x[i], :], np.copy(A[i, :])
        for j in range(i + 1, len(A)):
            A[j, i] = A[j, i] / A[i, i]
            A[j, i + 1:] = -1 * A[j, i] * A[i, i + 1:] + A[j, i + 1:]
    return A, x


def permutation(p, x):
    x = np.copy(x)
    for row, row_to_swap_with in reversed(list(enumerate(p))):
        x[row], x[row_to_swap_with] = x[row_to_swap_with], np.copy(x[row])
    return x


def vorwaerts(LU, x):
    x = np.copy(x)
    for row in range(len(LU)):
        x[row] = x[row] / LU[row, row]
        x[row + 1:] = x[row + 1:] - x[row] * LU[row, row + 1:]
    return x


def rueckwaerts(LU, x):
    x = np.copy(x)
    for row in reversed(range(len(LU))):
        x[:row] = x[:row] - x[row] * LU[row, :row]
    return x


if __name__ == '__main__':
    print('-' * 30)
    print('Exercise 1')
    print('-' * 30)

    # basic tests
    A = np.array([
        [0, 0, 0, 1],
        [2, 1, 2, 0],
        [4, 4, 0, 0],
        [2, 3, 1, 0],
    ], dtype=np.float64)
    LU, p = zerlegung_mit_pivot(A)
    b = np.array([152, 154, 56, 17], dtype=np.float64)

    print(f'input: {b}')
    b = vorwaerts(LU, b)
    b = rueckwaerts(LU, b)
    x = permutation(p, b)
    print(f'solution: {x}')
    print('-' * 30)
