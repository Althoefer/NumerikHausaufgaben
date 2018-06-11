# -*- coding: utf-8 -*-

import numpy as np


def integrate_trapezoid(f, a, b, slices):
    h = (b - a) / slices
    x = np.array([a + i * h for i in range(0, slices + 1)], dtype=np.float64)
    y = f(x)
    sum_ = y[0] + np.sum(2 * y[1:-1]) + y[-1]
    return (h / 2) * sum_


def integrate_simpson(f, a, b, slices):
    slices = 2 * slices
    h = (b - a) / slices
    x = np.array([a + i * h for i in range(0, slices + 1)], dtype=np.float64)
    y = f(x)
    sum_ = y[0] + np.sum(4 * y[1:-1:2]) + np.sum(2 * y[2:-1:2]) + y[-1]
    return (h / 3) * sum_


def integrate_gauss(f, a, b):
    x = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)], dtype=np.float64)
    beta = np.array([5 / 9, 8 / 9, 5 / 9], dtype=np.float64)
    
    x = x * (b - a) / 2 + (b + a) / 2
    beta = beta * (b - a) / 2
    return np.sum(beta * f(x))

def main():
    print('-' * 30)
    print('Numerical Integration')
    print('-' * 30)

    def f(x):
        return 1 / (1 + x ** 2)

    int_exact = np.arctan(1)
    print(f'exact: {int_exact}')
    print(f'error: {np.abs(int_exact - int_exact)}')
    print('-' * 30)

    int_trap = integrate_trapezoid(f, 0, 1, 8)
    print(f'trapezoids: {int_trap}')
    print(f'error: {np.abs(int_trap - int_exact)}')
    print('-' * 30)

    int_simp = integrate_simpson(f, 0, 1, 4)
    print(f'simpson: {int_simp}')
    print(f'error: {np.abs(int_simp - int_exact)}')
    print('-' * 30)

    int_gauss = integrate_gauss(f, 0, 1)
    print(f'gauss: {int_gauss}')
    print(f'error: {np.abs(int_gauss - int_exact)}')
    print('-' * 30)


if __name__ == '__main__':
    main()
