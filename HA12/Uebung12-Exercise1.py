# -*- coding: utf-8 -*-

import numpy as np


def gauss_newton(f_deriv, F, F_deriv, x0, epsilon=10 ** -4):
    x = np.copy(x0)
    while np.linalg.norm(f_deriv(x), ord=np.inf) > epsilon:
        x = x - np.linalg.pinv(F_deriv(x)).dot(F(x))
    return x

def levenberg_marquardt(f, f_deriv, F, F_deriv, x0, mu0, v0, epsilon=10 ** -4):
    x, mu, v = np.copy(x0), mu0, v0

    while np.linalg.norm(f_deriv(x), ord=np.inf) > epsilon:
        mat = np.transpose(F_deriv(x)).dot(F_deriv(x)) + mu * np.eye(len(x))
        rhs = -1 * np.transpose(F_deriv(x)).dot(F(x))
        p = np.linalg.solve(mat, rhs)
        
        x_new = x + p
        
        def q(y):
            vec = F(x) + F_deriv(x).dot(y - x)
            return 0.5 *  np.transpose(vec).dot(vec)
        
        gain = (f(x) - f(x_new)) / (q(x) - q(x_new))
        
        if gain > 0:
            x = x_new
            mu = mu * max(1 / 3, 1 - (2 * gain - 1) ** 3)
            v = 2
        else:
            mu = v * mu
            v = 2 * v
    
    return x


def main():
    print('-' * 30)
    print('Nichtlineare Ausgleichsprobleme')
    print('-' * 30)
        
    def F(x):
        x1, x2 = x
        ret1, ret2 = np.sin(x1) - x2, np.e ** (-1 * x2) - x1
        return np.array([ret1, ret2], dtype=np.float64)

    def F_deriv(x):
        x1, x2 = x
        ret11 = np.cos(x1)
        ret12 = -1
        ret22 = -1 * np.e ** (-1 * x2)
        return np.array([[ret11, ret12], [ret12, ret22]], dtype=np.float64)

    def f(x):
        return 0.5 * np.transpose(F(x)).dot(F(x))
    
    def f_deriv(x):
        return np.transpose(F_deriv(x)).dot(F(x))
        
    points = [
        np.array([5, 2], dtype=np.float64),
        np.array([6, 2], dtype=np.float64),
        np.array([-1, -1], dtype=np.float64),
    ]
    
    mu, v = 1, 1
    
    for point in points:
        print(f'Point: {point}')
        print('-' * 30)

        res_gauss = gauss_newton(f_deriv, F, F_deriv, point)
        print(f'gauss_newton: {res_gauss}')

        res_lev = levenberg_marquardt(f, f_deriv, F, F_deriv, point, mu, v)
        print(f'levenberg_marquardt: {res_lev}')
        print('-' * 30)


if __name__ == '__main__':
    main()
