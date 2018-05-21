# -*- coding: utf-8 -*-
import math

def newton(f, f_deriv, x0=0, value=0, iterations=20):
    x = x0
    for _ in range(iterations):
        x = x - (f(x) - value) / f_deriv(x)
    return x

def sekanten(f, x_minus_1, x_0=0, value=0, iterations=20):
    # start values
    x_k_minus_1, x_k = x_minus_1, x_0
    # storage for previously computed values of function f
    f_x_k_minus_1 = f(x_k_minus_1)

    for _ in range(iterations):
        # to avoid ZeroDivisionError
        if x_k == x_k_minus_1:
            break
        
        f_x_k = f(x_k)
        
        x_k_plus_1 = x_k - ((x_k - x_k_minus_1) / (f_x_k - f_x_k_minus_1)) * (f_x_k - value)

        # replace values for next iteration
        x_k_minus_1, x_k = x_k, x_k_plus_1
        f_x_k_minus_1 = f_x_k

    return x_k

def f(x):
    a = 9.8606
    c = -1.1085 * 10 ** 25
    d = 0.029
    return a / (1 - c * math.e ** (-1 * d * x))

def f_deriv(x):
    a = 9.8606
    c = -1.1085 * 10 ** 25
    d = 0.029
    return -1 * (a * c * d * math.e ** (-1 * d * x)) / (1 - c * math.e ** (-1 * d * x)) ** 2

if __name__ == '__main__':
    print('-' * 30)
    print('Nullstellen Verfahren')
    print('-' * 30)

    iterations = [i for i in range(0, 10)]

    columns = ['#Iterations', 'Newton-Approximation', 'Sekanten-Approximation']
    columns_length = [len(s) for s in columns]
    print(' '.join(columns))
    for iteration in iterations:
        x_newton = newton(f, f_deriv, 1961, value=9, iterations=iteration)
        x_sekanten = sekanten(f, 1961, 2000, value=9, iterations=iteration)
        print(f'{iteration:{columns_length[0]}d} {x_newton:{columns_length[1]}.15f} {x_sekanten:{columns_length[2]}.15f}')

    print('-' * 30)
        