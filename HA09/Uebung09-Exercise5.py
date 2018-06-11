# -*- coding: utf-8 -*-

import numpy as np
import scipy.integrate


def integrate_trapezoid(f, a, b, slices):
    h = (b - a) / slices
    x = np.array([a + i * h for i in range(0, slices + 1)], dtype=np.float64)
    y = f(x)
    sum_ = y[0] + np.sum(2 * y[1:-1]) + y[-1]
    return (h / 2) * sum_


def romberg(f, a, b, order):
    # choose minimum number of rows to compute for this order
    nrows = order // 2
    l = [(b - a) / (2 ** i) for i in range(nrows)]
    aitken_neville = [[integrate_trapezoid(f, a, b, 2 ** i)] for i in range(nrows)]
    for i in range(1, nrows):
        for j in range(1, i+1):
            nominator = aitken_neville[i][j-1] - aitken_neville[i-1][j-1]
            denominator = (l[i - j] / l[i]) ** 2 - 1
            aitken_neville[i].append(aitken_neville[i][j-1] + nominator / denominator)
    return aitken_neville[-1][-1]


def integrate_trapezoid2(f, a, b, slice_length):
    slices = int((b - a) / slice_length)
    h = (b - a) / slices
    x = np.array([a + i * h for i in range(0, slices + 1)], dtype=np.float64)
    y = f(x)
    sum_ = y[0] + np.sum(2 * y[1:-1]) + y[-1]
    return (h / 2) * sum_


def bulirsch(f, a, b, order):
    # choose minimum number of rows to compute for this order
    nrows = order // 2
    l = [b - a, (b - a) / 2, (b - a) / 3]
    while len(l) < nrows:
        l.append(l[len(l) - 2] / 2)
    aitken_neville = [[integrate_trapezoid2(f, a, b, l[i])] for i in range(nrows)]
    for i in range(1, nrows):
        for j in range(1, i+1):
            nominator = aitken_neville[i][j-1] - aitken_neville[i-1][j-1]
            denominator = (l[i - j] / l[i]) ** 2 - 1
            aitken_neville[i].append(aitken_neville[i][j-1] + nominator / denominator)
    return aitken_neville[-1][-1]


def main():
    print('-' * 30)
    print('Numerical Integration')
    print('-' * 30)

    print('basic test 1')
    print('-' * 30)

    a, b = 0, 1
    def f(x):
        return 1 / (1 + x ** 2)

    for order in range(2, 7, 2):
        int_rom = romberg(f, 0, 1, order=order)
        print(f'romberg (order={order}): {int_rom}')
    print('-' * 30)

    print('basic test 2')
    print('-' * 30)

    a, b = -2, 2
    def f(x):
        return (135 * x ** 2) / (2 + np.abs(x))
    
    for order in range(2, 7, 2):
        int_rom = romberg(f, -2, 2, order=order)
        print(f'romberg (order={order}): {int_rom}')
    print('-' * 30)

    for order in range(2, 7, 2):
        int_bul = bulirsch(f, -2, 2, order=order)
        print(f'bulirsch (order={order}): {int_bul}')
    print('-' * 30)

    print('basic test 3')
    print('-' * 30)

    a, b = 1, 2
    def f(x):
        return 1 / x
    
    int_quad, _ = scipy.integrate.quad(f, a, b)
    print(f'scipy.integrate.quad: {int_quad}')
    print('-' * 30)

    for order in range(2, 17, 2):
        int_rom = romberg(f, a, b, order=order)
        print(f'romberg (order={order}): {int_rom}')
        print(f'error: {np.abs(int_quad - int_rom)}')
    print('-' * 30)

    for order in range(2, 17, 2):
        int_bul = bulirsch(f, a, b, order=order)
        print(f'bulirsch (order={order}): {int_bul}')
        print(f'error: {np.abs(int_quad - int_bul)}')
    print('-' * 30)

    print('advanced test (exercise)')
    print('-' * 30)

    a, b = -1, 1
    def f(x):
        return np.sin(np.pi * x ** 2)

    int_quad, _ = scipy.integrate.quad(f, a, b)
    print(f'scipy.integrate.quad: {int_quad}')
    print('-' * 30)

    for order in range(2, 17, 2):
        int_rom = romberg(f, a, b, order=order)
        print(f'romberg (order={order}): {int_rom}')
        print(f'error: {np.abs(int_quad - int_rom)}')
    print('-' * 30)

    for order in range(2, 17, 2):
        int_bul = bulirsch(f, a, b, order=order)
        print(f'bulirsch (order={order}): {int_bul}')
        print(f'error: {np.abs(int_quad - int_bul)}')
    print('-' * 30)


if __name__ == '__main__':
    main()    
