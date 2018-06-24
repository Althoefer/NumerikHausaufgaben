# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as sciopt


def bfgs_quasi_newton(f, f_deriv, x0, alpha_start, rho, tau, epsilon=10 ** -4):
    x = np.copy(x0)
    D = np.eye(len(x))
    
    while np.linalg.norm(f_deriv(x), ord=np.inf) > epsilon:
        alpha = alpha_start
        p = -1 * np.dot(D, f_deriv(x))

        # determine alpha with armijo line search
        phi = f(x + alpha * p)
        lambda_ = f(x) + rho * np.dot(f_deriv(x), p) * alpha
        while phi > lambda_:
            alpha = tau * alpha

            phi = f(x + alpha * p)
            lambda_ = f(x) + rho * np.dot(f_deriv(x), p) * alpha

        x_new = x + alpha * p
        
        # conditional update on D
        h = x_new - x
        y = f_deriv(x_new) - f_deriv(x)
        v = np.dot(D, y)
        chi2 = 1 / (np.transpose(h).dot(y))
        chi1 = chi2 * (1 + chi2 * np.transpose(y).dot(v))
        
        if np.transpose(h).dot(y) > 0:
            D = D + chi1 * np.outer(h, np.transpose(h)) - chi2 * (np.outer(h, np.transpose(v)) + np.outer(v, np.transpose(h)))

        x = x_new

    return x


def main():
    print('-' * 30)
    print('BFGS-Quasi-Newton with Armijo Line Search')
    print('-' * 30)
    
    def f(x):
        x1, x2 = x
        return (np.sin(x1) - x2) ** 2 + (np.e ** (-1 * x2) - x1) ** 2
    
    def f_deriv(x):
        x1, x2 = x
        ret1 = 2 * np.cos(x1) * np.sin(x1) - 2 * x2 * np.cos(x1) - 2 * np.e ** (-1 * x2) + 2 * x1
        ret2 = -2 * np.sin(x1) + 2 * x2 - 2 * np.e ** (-2 * x2) + 2 * x1 * np.e ** (-1 * x2)
        return np.array([ret1, ret2], dtype=np.float64)

    alpha_start = 1
    rho, tau =  0.5, 0.5
    points = [
        np.array([5, 2], dtype=np.float64),
        np.array([6, 2], dtype=np.float64),
        np.array([-1, -1], dtype=np.float64),
        np.array([-2, -2], dtype=np.float64),
    ]

    for point in points:
        print(f'Point: {point}')
        print('-' * 30)

        res_bfgs = bfgs_quasi_newton(f, f_deriv, point, alpha_start, rho, tau)
        print(f'bfgs_quasi_newton: {res_bfgs}')
        print('-' * 30)

        res_scipy = sciopt.fmin(f, point)
        print(f'scipy.optimize.fmin min: {res_scipy}')
        print('-' * 30)

if __name__ == '__main__':
    main()
