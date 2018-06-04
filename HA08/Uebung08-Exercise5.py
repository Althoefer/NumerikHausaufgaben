# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def reconstruct_pseudoinverse(A, b):
    return np.dot(np.linalg.pinv(A), b)


def reconstruct_tsvd(A, b, alpha):
    u, s, vh = np.linalg.svd(A)
    s = np.where(max(s) / s > 1 / alpha, 0, s)
    # silence 0 division warning, because it is handled manually
    with np.errstate(divide='ignore'):
        s = np.where(s == 0, 0, 1 / s)
    return u.dot(np.diag(s)).dot(vh).dot(b)

def main():
    # for reproducible results
    np.random.seed(1234567890)

    print('-' * 30)
    print('Signal Reconstruction')
    print('-' * 30)
    
    def f(val):
        if 45 <= val and val <= 55:
            return 1
        if 60 <= val and val <= 65:
            return 0.5
        return 0
    f = np.vectorize(f, otypes=[np.float64])
    
    i = np.arange(0, 100, 1, dtype=np.float64)
    n = 100
    gamma = 0.05
    sigma = 10 ** -6

    A = np.empty((n, n), dtype=np.float64)
    for row in range(n):
        for col in range(n):
            c = 1 / (gamma * np.sqrt(2 * np.pi))
            expo = - ((row - col) / (np.sqrt(2) * n * gamma)) ** 2
            A[row, col] = (c / n) * np.e ** expo
    x = f(i)
    b = np.dot(A, x)
    # add noise
    b += sigma * np.random.randn(n)
    
    print('input')
    print('-' * 30)

    plt.plot(i, x)
    plt.plot(i, b)
    plt.legend(['original signal', 'smoothed signal'])
    plt.grid(True)
    plt.show()

    print('-' * 30)

    print('a) pseudoinverse')
    print('-' * 30)
    
    signal_pseudoinverse = reconstruct_pseudoinverse(A, b)

    plt.plot(i, b)
    plt.plot(i, signal_pseudoinverse)
    plt.legend(['smoothed signal', 'signal (pseudoinverse)'])
    plt.grid(True)
    plt.show()

    print('-' * 30)

    print('b) tsvd')
    print('-' * 30)
    
    for expo in range(0, -9, -1):
        signal_tsvd = reconstruct_tsvd(A, b, alpha=10 ** expo)
        plt.plot(i, x)
        plt.plot(i, signal_tsvd)
        plt.legend(['original signal', f'signal (tsvd alpha={10 ** expo})'])
        plt.grid(True)
        plt.show()

    print('-' * 30)


if __name__ == '__main__':
    main()
