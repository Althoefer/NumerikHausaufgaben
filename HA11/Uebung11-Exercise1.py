# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as sciopt


def newton_levenberg_marquardt(f, f_deriv, f_deriv2, x0, mu0, delta, epsilon=10 ** -4):
    x, mu = np.copy(x0), mu0
    
    while np.linalg.norm(f_deriv(x), ord=np.inf) > epsilon:
        mat = f_deriv2(x) + mu * np.eye(len(x))    
        while not np.all(np.linalg.eigh(mat)[0] > 0):
            mu = 2 * mu
            mat = f_deriv2(x) + mu * np.eye(len(x))
        
        p = np.linalg.solve(mat, -1 * f_deriv(x))
        x_new = x + p
        
        def q(y):
            return f(x) + f_deriv(x).dot(y - x) + 0.5 * np.transpose(y - x).dot(f_deriv2(x)).dot(y - x)
        gain = (f(x) - f(x_new)) / (q(x) - q(x_new))
    
        if gain > delta:
            x = x_new
            mu = mu * max(1/3, 1 - (2 * gain - 1) ** 3)
        else:
            mu = 2 * mu
    
    return x


def main():
    print('-' * 30)
    print('Newton with Levenberg-Marquardt')
    print('-' * 30)
    
    def f(x):
        x1, x2 = x
        return (np.sin(x1) - x2) ** 2 + (np.e ** (-1 * x2) - x1) ** 2
    
    def f_deriv(x):
        x1, x2 = x
        ret1 = 2 * np.cos(x1) * np.sin(x1) - 2 * x2 * np.cos(x1) - 2 * np.e ** (-1 * x2) + 2 * x1
        ret2 = -2 * np.sin(x1) + 2 * x2 - 2 * np.e ** (-2 * x2) + 2 * x1 * np.e ** (-1 * x2)
        return np.array([ret1, ret2], dtype=np.float64)

    def f_deriv2(x):
        x1, x2 = x
        ret11 = -2 * np.sin(x1) ** 2 + 2 * np.cos(x1) ** 2 + 2 * x2 * np.sin(x1) + 2
        ret12 = -2 * np.cos(x1) + 2 * np.e ** (-1 * x2)
        ret22 = 2 + 4 * np.e ** (-2 * x2) - 2 * x1 * np.e** (-1 * x2)
        return np.array([[ret11, ret12], [ret12, ret22]], dtype=np.float64)

    mu, delta = 1, 10 ** -3
    points = [
        np.array([5, 2], dtype=np.float64),
        np.array([6, 2], dtype=np.float64),
        np.array([-1, -1], dtype=np.float64),
        np.array([-2, -2], dtype=np.float64),
    ]

    for point in points:
        print(f'Point: {point}')
        print('-' * 30)

        res_new = newton_levenberg_marquardt(f, f_deriv, f_deriv2, point, mu, delta)
        print(f'newton_levenberg_marquardt: {res_new}')
        print('-' * 30)

        res_scipy = sciopt.fmin(f, point)
        print(f'scipy.optimize.fmin min: {res_scipy}')
        print('-' * 30)

if __name__ == '__main__':
    main()
