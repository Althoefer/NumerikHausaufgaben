# -*- coding: utf-8 -*-
import numpy as np

def newton(f, f_deriv, x0=0, value=0, iterations=20):
    x = x0
    for _ in range(iterations):
        f_x = f(x) - value
        f_deriv_x = f_deriv(x)
        if f_x == 0 or f_deriv_x == 0:
            break
        x = x - f_x / f_deriv_x
    return x

def newton_mod1(f, f_deriv, q, x0=0, value=0, iterations=20):
    x = x0
    for _ in range(iterations):
        f_x = f(x) - value
        f_deriv_x = f_deriv(x)
        if f_x == 0 or f_deriv_x == 0:
            break
        x = x - q * f_x / f_deriv_x
    return x

def newton_mod2(f, f_deriv, f_deriv2, x0=0, value=0, iterations=20):
    x = x0
    for _ in range(iterations):
        f_x = f(x) - value
        f_deriv_x = f_deriv(x)
        f_deriv2_x = f_deriv2(x)
        nominator = f_x * f_deriv_x
        denominator = f_deriv_x ** 2 - f_x * f_deriv2_x
        if nominator == 0 or denominator == 0:
            break
        x = x - nominator / denominator
    return x

def f(x):
    return np.arctan(x) - x

def f_deriv(x):
    return 1 / (1 + x ** 2) - 1

def f_deriv2(x):
    return -2 * x / (1 + x ** 2) ** 2

if __name__ == '__main__':
    print('-' * 30)
    print('Newton Verfahren mit mehrfachen Nullstellen')
    print('-' * 30)

    iterations = [i for i in range(0, 20)]

    columns = ['#Iterations', 'Newton-Approximation', 'Mod1-Approximation', 'Mod2-Approximation']
    columns_length = [len(s) for s in columns]
    print(' '.join(columns))
    x0 = 1
    for iteration in iterations:
        x_newton = newton(f, f_deriv, x0=x0, iterations=iteration)
        x_newton_mod1 = newton_mod1(f, f_deriv, q=3, x0=x0, iterations=iteration)
        x_newton_mod2 = newton_mod2(f, f_deriv, f_deriv2, x0=x0, iterations=iteration)
        print(f'{iteration:{columns_length[0]}d} {x_newton:{columns_length[1]}.15f} {x_newton_mod1:{columns_length[2]}.15f} {x_newton_mod2:{columns_length[3]}.15f}')

    print('-' * 30)
