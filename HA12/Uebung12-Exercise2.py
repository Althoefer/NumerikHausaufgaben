# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def gauss_newton(f_deriv, F, F_deriv, x0, epsilon=10 ** -4):
    x = x0
    while f_deriv(x) > epsilon:
        x = x - float(np.linalg.pinv(F_deriv(x)).dot(F(x)))
    return x


def main():
    print('-' * 30)
    print('Nichtlineare Ausgleichsprobleme')
    print('-' * 30)

    print('a) Least Squares')
    print('-' * 30)

    x = np.array([(2 * i) / 5 - 1 for i in range(6)], dtype=np.float64)
    y = np.cos(np.pi * x)
    A = np.transpose(np.array([x ** i for i in range(3)], dtype=np.float64))
    
    res_normal = np.linalg.solve(A.T.dot(A), A.T.dot(y))
    print(f'coeffs with normal equation: {res_normal}')
    print('-' * 30)

    p = res_normal
    def g(x):
        p0, p1, p2 = p
        return p0 + p1 * x + p2 * x ** 2

    def g_deriv(x):
        p0, p1, p2 = p
        return p1 + 2 * p2 * x
    
    x_input = np.arange(-1.5, 1.5, 0.001, dtype=np.float64)
    y_output = np.array([
        g(i) for i in x_input
    ], dtype=np.float64)

    plt.scatter(x, y)
    plt.plot(x_input, y_output)
    plt.show()
    print('-' * 30)

    print('b) Least Squares Distance')
    print('-' * 30)

    sum_ = 0
    for x_i, y_i in zip(x, y):
        
        def F(h):
            return np.sqrt(2) * np.array([g(x_i + h) - y_i, h], dtype=np.float64)

        def F_deriv(h):
            return np.sqrt(2) * np.array([
                [g_deriv(x_i + h)],
                [1],
            ], dtype=np.float64)
    
        def f(h):
            return 0.5 * np.transpose(F(h)).dot(F(h))
        
        def f_deriv(h):
            return np.transpose(F_deriv(h)).dot(F(h))
    
        point = 0
        res_gauss = gauss_newton(f_deriv, F, F_deriv, point)
        sum_ += f(res_gauss)
        print(f'distance with gauss_newton: d({res_gauss}) ** 2 = {f(res_gauss)}')
        print('-' * 30)
    sum_ = 0.5 * sum_
    print(f'sum of squared distances: {sum_}')
    print('-' * 30)

    print('c) Least Squares Distance')
    print('-' * 30)

    def g(p, x):
        return p[0] + p[1] * x + p[2] * x ** 2

    def g_deriv(p, x):
        return p[1] + 2 * p[2] * x

    def F(p, h):
        return np.array([
            g(p, x[0] + h[0]) - y[0],
            g(p, x[1] + h[1]) - y[1],
            g(p, x[2] + h[2]) - y[2],
            h[0],
            h[1],
            h[2],
        ], dtype=np.float64)

    def F_deriv(p, h):
        J = [
            [0, 1, 2 * (x[0] + h[0]), ],
            [],
            [],
            [],
            [],
            [],
        ]
        return np.sqrt(2) * np.array([
            [g_deriv(x_i + h)],
            [1],
        ], dtype=np.float64)

    def f(p, h):
        return 0.5 * F(p, h).T.dot(F(p, h))

    def f_deriv(p, h):
        return F_deriv(p, h).T.dot(F(p, h))


if __name__ == '__main__':
    main()
