# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def interpolate_newton(points):
    npoints = len(points)

    # build up diffs as lower triangular matrix
    diffs = [[points[i][1]] for i in range(npoints)]
    for i in range(1, npoints):
        for j in range(1, i+1):
            nominator = diffs[i][j-1] - diffs[i-1][j-1]
            denominator = points[i][0] - points[i-j][0]
            diffs[i].append(nominator / denominator)

    def horner_evaluation(x):
        q = diffs[-1][-1]
        for i in reversed(range(npoints-1)):
            q = diffs[i][-1] + (x - points[i][0]) * q
        return q

    return horner_evaluation


if __name__ == '__main__':
    print('-' * 30)
    print('Newton Interpolation')
    print('-' * 30)

    print('a) basic tests')
    print('-' * 30)

    points = [(0, 3), (1, 2), (3, 6)]
    x = 2
    solu = 3

    p = interpolate_newton(points)
    y = p(x)

    print(f'interpolating points: {points}')
    print(f'p({x}) = {y} <= {y == solu} => {solu}')
    print('-' * 30)

    print('b) advanced tests')
    print('-' * 30)

    func = lambda x: 1 / (1 + x ** 2)
    ms = [7, 9, 11]
    polynoms = []
    
    for m in ms:
        def get_x(i):
            return -5 + 10 * i / (m - 1)
        points = [(get_x(i), func(get_x(i))) for i in range(0, m)]

        p = interpolate_newton(points)
        polynoms.append(p)

    x = np.arange(-5, 5, 0.01)
    plt.plot(x, func(x))
    legend = ['$f(x)$']
    for m, p in zip(ms, polynoms):
        plt.plot(x, p(x))
        legend.append('$p_{' + str(m) + '}(x)$')
    plt.legend(legend)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'Newton Interpolation for $f(x) = \frac{1}{1 + x^2}$')
    plt.grid(True)
    plt.show()
    print('-' * 30)

    print('c) advanced tests')
    print('-' * 30)
    
    func = lambda x: 1 / (1 + x ** 2)
    ms = [7, 9, 11]
    polynoms = []
    
    for m in ms:
        def get_x(i):
            return -5 * np.cos(np.pi * (2 * i + 1) / (2 * m))
        points = [(get_x(i), func(get_x(i))) for i in range(0, m)]

        p = interpolate_newton(points)
        polynoms.append(p)

    x = np.arange(-5, 5, 0.01)
    plt.plot(x, func(x))
    legend = ['$f(x)$']
    for m, p in zip(ms, polynoms):
        plt.plot(x, p(x))
        legend.append('$p_{' + str(m) + '}(x)$')
    plt.legend(legend)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'Newton Interpolation for $f(x) = \frac{1}{1 + x^2}$')
    plt.grid(True)
    plt.show()
    print('-' * 30)

    print('d) advanced tests')
    print('-' * 30)
    
    func_1 = lambda x: 1 / (1 + x) * np.cos(3 * np.pi * x)
    func_2 = lambda x: 1 / (1 + x) * np.sin(3 * np.pi * x)
    ms = [6, 7, 8]
    polynoms = []
    
    for m in ms:
        def get_x(i):
            return i / (m - 1)
        points_1 = [(get_x(i), func_1(get_x(i))) for i in range(0, m)]
        points_2 = [(get_x(i), func_2(get_x(i))) for i in range(0, m)]
        
        p_1 = interpolate_newton(points_1)
        polynoms.append(p_1)
        p_2 = interpolate_newton(points_2)
        polynoms.append(p_2)

    x = np.arange(0, 1, 0.01)
    plt.plot(func_1(x), func_2(x))
    legend = [r'$\gamma(t)$']
    for m, index in zip(ms, range(0, len(polynoms), 2)):
        plt.plot(polynoms[index](x), polynoms[index+1](x))
        legend.append('$p_{' + str(m) + '}(t)$')
    plt.legend(legend)
    plt.xlabel('x')
    plt.ylabel('y')
    # \binom as workaround for missing vectors in matplotlib's mathtext
    plt.title(r'Newton Interpolation for $\gamma(t) = \frac{1}{1 + t} \binom{\cos(3 \pi t)}{\sin(3 \pi t)}$')
    plt.grid(True)
    plt.show()
    print('-' * 30)
