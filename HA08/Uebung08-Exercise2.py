# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def approx_normal(x_points, y_points, degree=1):
    assert len(x_points) == len(y_points)
    assert degree >= 0

    npoints = len(x_points)

    A = np.empty((npoints, degree + 1), dtype=np.float64)
    for i in range(degree + 1):
        A[:, i] = x_points ** i
    b = np.copy(y_points)

    mat = np.dot(np.transpose(A), A)
    rhs = np.dot(np.transpose(A), b)
    coeffs = np.linalg.solve(mat, rhs)

    def horner(x):
        res = coeffs[-1]
        for i in reversed(range(degree)):
            res = coeffs[i] + x * res
        return res

    return horner

def approx_qr(x_points, y_points, degree=1):
    assert len(x_points) == len(y_points)
    assert degree >= 0

    npoints = len(x_points)

    A = np.empty((npoints, degree + 1), dtype=np.float64)
    for i in range(degree + 1):
        A[:, i] = x_points ** i
    b = np.copy(y_points)

    Q, R = np.linalg.qr(A)
    c = np.dot(np.transpose(Q), b)
    coeffs = np.linalg.solve(R, c)

    def horner(x):
        res = coeffs[-1]
        for i in reversed(range(degree)):
            res = coeffs[i] + x * res
        return res

    return horner

def main():
    print('-' * 30)
    print('Ausgleichspolynome')
    print('-' * 30)

    print('0) basic test')
    print('-' * 30)

    x_points = np.array([0, 2, 3], dtype=np.float64)
    y_points = np.array([1, 2, 2], dtype=np.float64)
    x = np.arange(-1, 5, 0.01, dtype=np.float64)

    p_normal = approx_normal(x_points, y_points, degree=1)
    p_qr = approx_qr(x_points, y_points, degree=1)

    plt.plot(x, p_normal(x))
    plt.plot(x, p_qr(x))
    plt.scatter(x_points, y_points)
    plt.legend(['$p_{normal}(x)$', '$p_{qr}(x)$'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Basic Test for linear Approximation')
    plt.grid(True)
    plt.show()
    print('-' * 30)

    print('a) normal')
    print('-' * 30)

    func = lambda x: x - x ** 3
    epsilon = 10 ** -4
    x_points = np.array([
        0, epsilon, 2 * epsilon, 3 * epsilon,
        1 - 3 * epsilon, 1 - 2 * epsilon, 1 - epsilon, 1,
    ], dtype=np.float64)
    y_points = func(x_points)
    x = np.arange(-5, 5, 0.01, dtype=np.float64)

    p_normal = approx_normal(x_points, y_points, degree=5)

    plt.plot(x, func(x))
    plt.plot(x, p_normal(x))
    plt.scatter(x_points, y_points)
    plt.legend(['$f(x)$', '$p_{normal}(x)$'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Approximation of $f(x) = x - x^3$ by using the Normal Equation')
    plt.grid(True)
    plt.show()

    print('-' * 30)

    print('b) qr')
    print('-' * 30)

    func = lambda x: x - x ** 3
    epsilon = 10 ** -4
    x_points = np.array([
        0, epsilon, 2 * epsilon, 3 * epsilon,
        1 - 3 * epsilon, 1 - 2 * epsilon, 1 - epsilon, 1,
    ], dtype=np.float64)
    y_points = func(x_points)
    x = np.arange(-5, 5, 0.01, dtype=np.float64)

    p_qr = approx_qr(x_points, y_points, degree=5)

    plt.plot(x, func(x))
    plt.plot(x, p_qr(x))
    plt.scatter(x_points, y_points)
    plt.legend(['$f(x)$', '$p_{qr}(x)$'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Approximation of $f(x) = x - x^3$ by using the $QR$ Decomposition')
    plt.grid(True)
    plt.show()

    print('-' * 30)

if __name__ == '__main__':
    main()
