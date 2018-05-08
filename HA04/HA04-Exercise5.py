# -*- coding: utf-8 -*-

# Exercise 5
import numpy as np


####################
#  Start LU
def zerlegung_mit_pivot(A):
    A = np.copy(A)
    x = np.array([i for i in range(len(A))], dtype=np.int)
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
    for row, row_to_swap_with in enumerate(p):
        x[row], x[row_to_swap_with] = x[row_to_swap_with], np.copy(x[row])
    return x


def vorwaerts(LU, x):
    x = np.copy(x)
    for row in range(len(LU)):
        x[row + 1:] = x[row + 1:] - x[row] * LU[row + 1:, row]
    return x


def rueckwaerts(LU, x):
    x = np.copy(x)
    for row in reversed(range(len(LU))):
        x[row] = x[row] / LU[row, row]
        x[:row] = x[:row] - x[row] * LU[:row, row]
    return x


# End LU
####################

####################
# Start Householder
def householder(A):
    QR = np.copy(A)
    diagonal = np.array([], dtype=np.float64)
    for row in range(len(QR)):
        a = QR[row:, row]
        d = -1 * np.sign(a[0]) * np.linalg.norm(a)
        v = a + np.sign(a[0]) * np.linalg.norm(a) * np.array([1] + [0 for _ in range(len(a) - 1)], dtype=np.float64)
        v[0] = a[0] - d
        Q = np.eye(len(a), dtype=np.float64) + (1 / (v[0] * d)) * np.outer(v, np.transpose(v))
        QR[row:, row:] = np.dot(Q, QR[row:, row:])
        diagonal = np.append(diagonal, d)
        QR[row:, row] = v
    return QR, diagonal


def reconstruct_q(QR, diagonal, b):
    Q_total = np.eye(len(QR), dtype=np.float64)
    for row in range(len(QR)):
        Q = np.eye(len(QR), dtype=np.float64)
        v = QR[row:, row]
        Q[row:, row:] = Q[row:, row:] + (1 / (v[0] * diagonal[row])) * np.outer(v, np.transpose(v))
        Q_total = np.dot(Q, Q_total)
    return np.dot(Q_total, b)


def solve_Ry(QR, y, diagonal):
    x = np.copy(y)
    for row in reversed(range(len(QR))):
        x[row] = x[row] / diagonal[row]
        x[:row] = x[:row] - x[row] * QR[:row, row]
    return x


# End Householder
####################


# Tests
def main():
    print('-' * 30)
    print('Householder basic tests')
    print('-' * 30)

    A = np.array([
        [20, 18, 44],
        [0, 40, 45],
        [-15, 24, -108],
    ], dtype=np.float64)
    QR, diagonal = householder(A)

    b = np.array([-4, -45, 78], dtype=np.float64)
    y = reconstruct_q(QR, diagonal, b)
    x = solve_Ry(QR, y, diagonal)

    print('data for A * x = b')
    print('A')
    print(A)
    print('b')
    print(b)
    print('QR')
    print(QR)
    print('diagonal')
    print(diagonal)
    print('y')
    print(y)
    print('x')
    print(x)
    print('-' * 30)

    # basic tests
    print('-' * 30)
    print('Householder advanced tests')
    print('-' * 30)

    dims = [40, 50, 60]
    for n in dims:
        print(f'dimension: {n}')

        A = np.eye(n, dtype=np.float64)
        A[:, -1] = np.array([1 for _ in range(n)], dtype=np.float64)
        for i in range(n):
            A[i + 1:, i] = -1

        # zerlegungen
        QR, diagonal = householder(A)
        LU, p = zerlegung_mit_pivot(A)

        b = np.array([3 - i for i in range(1, n)] + [2 - n], dtype=np.float64)
        print(f'input: {b}')

        # solve LGS
        y = permutation(p, b)
        y = vorwaerts(LU, y)
        x = rueckwaerts(LU, y)
        print(f'solution LU w. pivot: {x}')
        y = reconstruct_q(QR, diagonal, b)
        x = solve_Ry(QR, y, diagonal)
        print(f'solution householder: {x}')
        print('-' * 30)


if __name__ == '__main__':
    main()
