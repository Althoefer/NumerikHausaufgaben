# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def gauss_newton(f_deriv, F, F_deriv, x0, epsilon=10 ** -4):
    x = x0
    while f_deriv(x) > epsilon:
        x = x - float(np.linalg.pinv(F_deriv(x)).dot(F(x)))
    return x


def gauss_newton2(f_deriv, F, F_deriv, x0, epsilon=10 ** -4):
    x = np.copy(x0)
    while np.linalg.norm(f_deriv(x), ord=np.inf) > epsilon:
        x = x - np.linalg.pinv(F_deriv(x)).dot(F(x))
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

    # args as combined vector of p and h
    def F(arg):
        p, h = arg[0:3], arg[3:9]
        return np.array([
            g(p, x[0] + h[0]) - y[0],
            g(p, x[1] + h[1]) - y[1],
            g(p, x[2] + h[2]) - y[2],
            g(p, x[3] + h[3]) - y[3],
            g(p, x[4] + h[4]) - y[4],
            g(p, x[5] + h[5]) - y[5],
            h[0],
            h[1],
            h[2],
            h[3],
            h[4],
            h[5],
        ], dtype=np.float64)

    # first columns: p-derivatives, later columns: h-derivatives
    def F_deriv(arg):
        p, h = arg[0:3], arg[3:9]
        J = [
            [1, x[0] + h[0], (x[0] + h[0]) ** 2, p[1] + 2 * p[2] * h[0], 0, 0, 0, 0, 0],
            [1, x[1] + h[1], (x[1] + h[1]) ** 2, 0, p[1] + 2 * p[2] * h[1], 0, 0, 0, 0],
            [1, x[2] + h[2], (x[2] + h[2]) ** 2, 0, 0, p[1] + 2 * p[2] * h[2], 0, 0, 0],
            [1, x[3] + h[3], (x[3] + h[3]) ** 2, 0, 0, 0, p[1] + 2 * p[2] * h[3], 0, 0],
            [1, x[4] + h[4], (x[4] + h[4]) ** 2, 0, 0, 0, 0, p[1] + 2 * p[2] * h[4], 0],
            [1, x[5] + h[5], (x[5] + h[5]) ** 2, 0, 0, 0, 0, 0, p[1] + 2 * p[2] * h[5]],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
        return np.array(J, dtype=np.float64)

    def f(arg):
        return 0.5 * F(arg).T.dot(F(arg))

    def f_deriv(arg):
        return F_deriv(arg).T.dot(F(arg))

    point = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    res_gauss = gauss_newton2(f_deriv, F, F_deriv, point)
    print(f'gauss_newton:')
    print(f'==> coeffs {res_gauss[0:3]}')
    print(f'==> distances {res_gauss[4:12]} ==> sum of squares: {np.sum(res_gauss[4:12] ** 2)}')
    x_input = np.arange(-1.5, 1.5, 0.001, dtype=np.float64)
    y_output = np.array([
        g(res_gauss[0:3], i) for i in x_input
    ], dtype=np.float64)

    plt.scatter(x, y)
    plt.plot(x_input, y_output)
    plt.show()

if __name__ == '__main__':
    main()
