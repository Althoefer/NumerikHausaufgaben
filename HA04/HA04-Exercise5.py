# -*- coding: utf-8 -*-

# Exercise 5
import numpy as np


####################
#  Start LU
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
    diagonal = np.array([])
    for row in range(len(QR) - 1):
        a = QR[row:, row]
        v = a + np.sign(a[0]) * np.linalg.norm(a) * np.array([1] + [0 for _ in range(len(a) - 1)], dtype=np.float64)
        Q = np.eye(len(a)) - (2 / np.dot(v, v)) * np.outer(v, np.transpose(v))
        QR[row:, row:] = np.dot(Q, QR[row:, row:])
        diagonal = np.append(diagonal, QR[row, row])
        QR[row:, row] = v
    diagonal = np.append(diagonal, QR[-1, -1])
    return QR, diagonal


def reconstruct_q(QR, diagonal, b):
    Q_total = np.eye(len(QR))
    for row in range(len(A)):
        Q = np.eye(len(QR))
        d = -1 * np.sign(QR[row, row]) * np.linalg.norm(QR[row:, row])
        v1 = QR[row, row] - d
        Q[row:, row:] = np.eye(len(Q[row:, row:])) - (2 / (-2 * v1 * d)) * np.outer(QR[row:, row], np.transpose(QR[row:, row]))
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
if __name__ == '__main__':
    print('-' * 30)
    print('Householder basic tests')
    print('-' * 30)

    A = np.array([
        [20, 18, 44],
        [0, 40, 45],
        [-15, 24, -108],
    ], dtype=np.float64)
    b = np.array([-4, -45, 78], dtype=np.float64)
    QR, diagonal = householder(A)
    # TODO: reconstruction of Q not working yet
    y = reconstruct_q(QR, diagonal, b)
    y = [50, 0, 75]
    x = solve_Ry(QR, y, diagonal)

    print('QR')
    print(QR)
    print('diagonal')
    print(diagonal)
    print('y')
    print(y)
    print('x')
    print(x)
    """
    # basic tests
    A = np.array([
        [0, 0, 0, 1],
        [2, 1, 2, 0],
        [4, 4, 0, 0],
        [2, 3, 1, 0],
    ], dtype=np.float64)
    rhs = [
        np.array([3, 5, 4, 5], dtype=np.float64),
        np.array([4, 10, 12, 11], dtype=np.float64),
    ]
    for b in rhs:
        Q, R = scipy.linalg.qr(A)
        print(Q)
        print(R)
        print(f'input: {A}')
        diagonal = []
        QR, diagonal = householder(A)
        print(f'solution: {QR}')
        print('-' * 30)

    # advanced tests
    print('-' * 30)
    print('Householder advanced tests')
    print('-' * 30)
    """
