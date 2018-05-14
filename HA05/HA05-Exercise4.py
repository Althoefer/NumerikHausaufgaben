# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from Blatt05_lib import Ablock, ew_exakt, plotev, animev

def my_sign(x):
    if x >= 0:
        return 1
    else:
        return -1
    
def get_square_sum(A):
    # sum of squares of A without the main diagonal
    A = np.copy(A)
    np.fill_diagonal(A, 0)
    return np.sum(A ** 2)

def jacobi(A, epsilon = 10 ** -3):
    A = np.copy(A)
    n = len(A)
    Q = np.eye(n)
    while get_square_sum(A) >= epsilon:
        max_index_x = -1
        max_index_y = -1
        max_value = -1
        for i in range(n-1):
            cur_max_index_x = i
            cur_max_index_y = i + 1 + np.argmax(abs(A[i, i+1:]))
            cur_max_value = abs(A[cur_max_index_x, cur_max_index_y])
            if max_value < cur_max_value:
                max_index_x = cur_max_index_x
                max_index_y = cur_max_index_y
                max_value = cur_max_value
        Qk = np.eye(n)
        alpha = (A[max_index_y, max_index_y] - A[max_index_x, max_index_x]) / (2 * A[max_index_x, max_index_y])
        c = sqrt(0.5 + 0.5 * sqrt(alpha ** 2 / (1 + alpha ** 2)))
        s =  my_sign(alpha) / (2 * c * sqrt(1 + alpha ** 2))
        Qk[max_index_x, max_index_x] = c
        Qk[max_index_y, max_index_y] = c
        Qk[max_index_x, max_index_y] = s
        Qk[max_index_y, max_index_x] = -1 * s

        A = np.dot(np.dot(np.transpose(Qk), A), Qk)
        Q = np.dot(Q, Qk)
    
    # sort ascending by eigenvalues
    v = np.diag(A)
    order = np.argsort(v)
    v = v[order]
    Q = Q[:, order]

    return v, Q

if __name__ == '__main__':
    print('-' * 30)
    print('Jacobi')
    print('-' * 30)
    
    print('basic tests')
    print('-' * 30)
    
    A = np.array([
            [1, 1 / 2, 1 / 3],
            [1 / 2, 1 / 3, 1 / 4],
            [1 / 3, 1 / 4, 1 / 5],
    ], dtype = np.float64)
    print('input matrix:')
    print(A)

    v, Q = jacobi(A)
    print('eigenvalues:')
    print(v)
    print('eigenvectors (columnwise):')
    print(Q)
    print('-' * 30)
    


    print('advanced tests')
    print('-' * 30)

    m = 10
    A = Ablock(m)
    print(f'running with: m = {m}')

    print('exact eigenvalues:')
    v = ew_exakt(m)
    plt.plot([i for i in range(len(v))], v)
    plt.xlabel('place of eigenvalue (in ordered list)')
    plt.ylabel('value of eigenvalue')
    plt.show()
    plt.clf()

    v, Q = jacobi(A)
    print('self-calculated eigenvalues:')
    plt.plot([i for i in range(len(v))], v)
    plt.xlabel('place of eigenvalue (in ordered list)')
    plt.ylabel('value of eigenvalue')
    plt.show()
    plt.clf()

    print('eigenvectors for 4 smallest eigenvalues:')
    for i in range(4):
        print('plot:')
        plotev(Q[:, i])
        plt.show()
        plt.clf()

        print('animation:')
        anim = animev(Q[:, i])()
        plt.show()
        plt.clf()
    
    print('-' * 30)
