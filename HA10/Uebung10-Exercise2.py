# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize as sciopt


def steepdest_descent(f, f_deriv, start_point, alpha_start, rho, tau, epsilon):
    x = start_point

    while np.linalg.norm(f_deriv(x), ord=np.inf) > epsilon:
        alpha = alpha_start
        p = -1 * f_deriv(x)

        phi = f(x + alpha * p)
        lambda_ = f(x) + rho * np.dot(f_deriv(x), p) * alpha
        while phi > lambda_:
            alpha = tau * alpha

            phi = f(x + alpha * p)
            lambda_ = f(x) + rho * np.dot(f_deriv(x), p) * alpha

        x = x + alpha * p
    
    return x

def main():
    print('-' * 30)
    print('Steepest-Descent')
    print('-' * 30)

    def f(x):
        x_1, x_2 = x
        return (np.sin(x_1) - x_2) ** 2 + (np.e ** (-1 * x_2) - x_1) ** 2

    def f_deriv(x):
        x_1, x_2 = x
        ret_1 = 2 * np.sin(x_1) * np.cos(x_1) - 2 * x_2 * np.cos(x_1) - 2 * np.e ** (-1 * x_2) + 2 * x_1
        ret_2 = -2 * np.sin(x_1) + 2 * x_2 - 2 * np.e ** (-2 * x_2) + 2 * x_1 * np.e ** (-1 * x_2)
        return np.array([ret_1, ret_2], dtype=np.float64)
    
    alpha_start = 1
    rho, tau = 0.5, 0.5
    epsilon = 10 ** -4
    
    start_points = [
        np.array([5, 2], dtype=np.float64),
        np.array([6, 2], dtype=np.float64),
        np.array([-1, -1], dtype=np.float64),
        np.array([-2, -2], dtype=np.float64),
    ]

    for start_point in start_points:
        print('-' * 30)
        print(f'start_point: {start_point}')
        print('-' * 30)

        res_steepdesc = steepdest_descent(f, f_deriv, start_point, alpha_start, rho, tau, epsilon)
        print(f'steepest descent min: {res_steepdesc}')
        print(f'steepest descent f: {f(res_steepdesc)}')
        print(f'steepest descent f_deriv: {f_deriv(res_steepdesc)}')
        print('-' * 30)

        res_scipy = sciopt.fmin(f, np.array(start_point))
        print(f'scipy.optimize.fmin min: {res_scipy}')
        print(f'scipy.optimize.fmin f: {f(res_scipy)}')
        print(f'scipy.optimize.fmin f_deriv: {f_deriv(res_scipy)}')
        print('-' * 30)


if __name__ == '__main__':
    main()
