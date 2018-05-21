# -*- coding: utf-8 -*-
import math

def newton_approx(f, f_deriv, alpha, x0=0, value=0, epsilon=10 ** -6):
    x_k = x0
    x_k_minus_1 = x_k - (f(x_k) - value) / f_deriv(x_k)
    while (alpha / (1 - alpha)) * abs(x_k - x_k_minus_1) >= epsilon:
        x_k_minus_1, x_k = x_k, x_k - (f(x_k) - value) / f_deriv(x_k)
    return x_k

def f(x):
    return x + math.log(x) - 2

def f_deriv(x):
    return 1 + 1 / x

import scipy.optimize
    
if __name__ == '__main__':
    print('-' * 30)
    print('Newton mit a-posteriori Fehlerabschätzung')
    print('-' * 30)

    x = newton_approx(f, f_deriv, alpha=0.25, x0=1)
    print(f'Nullstelle: x = {x}')
    print(f'Test: f({x}) = {f(x)}')

    print('-' * 30)
        