# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def householder(A):
    QR = np.copy(A)
    diagonal = np.array([], dtype=np.float64)
    for row in range(len(QR)):
        a = QR[row:, row]
        d = -1 * np.sign(a[0]) * np.linalg.norm(a)
        v = a + np.sign(a[0]) * np.linalg.norm(a) * np.array([1] + [0 for _ in range(len(a) - 1)], dtype=np.float64)
        v[0] = a[0] - d
        Q = np.eye(len(a), dtype=np.float64) + (1 / (v[0] * d)) * np.outer(v, np.transpose(v))
        QR[row:, row:] = np.dot(Q, QR[row:, row:])
        diagonal = np.append(diagonal, d)
        QR[row:, row] = v
    return QR, diagonal


def reconstruct_q(QR, diagonal, b):
    Q_total = np.eye(len(QR), dtype=np.float64)
    for row in range(len(QR)):
        Q = np.eye(len(QR), dtype=np.float64)
        v = QR[row:, row]
        Q[row:, row:] = Q[row:, row:] + (1 / (v[0] * diagonal[row])) * np.outer(v, np.transpose(v))
        Q_total = np.dot(Q, Q_total)
    return np.dot(Q_total, b)


def solve_Ry(QR, y, diagonal):
    x = np.copy(y)
    for row in reversed(range(len(QR))):
        x[row] = x[row] / diagonal[row]
        x[:row] = x[:row] - x[row] * QR[:row, row]
    return x


def interpolate_natural_spline(points):
    npoints = len(points)
    x_points, y_points = zip(*points)

    h = np.array([x_points[i + 1] - x_points[i] for i in range(npoints - 1)],
                  dtype=np.float64)
    
    A = np.diag(2 * (h[:-1] + h[1:])) \
            + np.diag(h[1:-1], k=-1) + np.diag(h[1:-1], k=1)
    
    b = np.array([0 for _ in range(npoints - 2)], dtype=np.float64)
    for row in range(1, npoints - 1):
        b[row-1] = 6 * ((y_points[row + 1] - y_points[row]) / h[row] \
                         - (y_points[row] - y_points[row - 1]) / h[row - 1])

    beta = np.array([0 for _ in range(npoints)], dtype=np.float64)
    
    QR, diagonal = householder(A)
    temp = reconstruct_q(QR, diagonal, b)
    beta[1:-1] = solve_Ry(QR, temp, diagonal)

    alpha = np.array([0 for _ in range(npoints-1)], dtype=np.float64)
    for row in range(npoints-1):
        alpha[row] = (y_points[row+1] - y_points[row]) / h[row] \
                        - (1 / 3) * beta[row] * h[row] \
                        - (1 / 6) * beta[row + 1] * h[row]
        
    def horner_evaluation(x):
        i = 0
        while x > x_points[i + 1] and i < npoints - 2:
            i += 1

        # coefficients of polynom from highest x degree to lowest
        coeffs = [
                (beta[i + 1] - beta[i]) / (6 * h[i]),
                (beta[i] / 2),
                alpha[i],
                y_points[i]
        ]
        q = 0
        for coeff in coeffs:
            q = coeff + (x - x_points[i]) * q
        return q
    
    return horner_evaluation


if __name__ == '__main__':
    print('-' * 30)
    print('Natural Cubic Splines')
    print('-' * 30)

    # additional example from script
    print('0) basic tests')
    print('-' * 30)

    points = [(-3, 3 / 5), (-1, 1), (0, 3 / 2), (1, 1), (3, 3 / 5)]
    x = 0
    solu = 3 / 2

    p = interpolate_natural_spline(points)
    y = p(x)

    print(f'points: {points}')
    print(f's({x}) = {y} <= {y == solu} => {solu}')
    print('-' * 30)

    print('a) basic tests (splines produce different result than newton interpolation)')
    print('-' * 30)

    points = [(0, 3), (1, 2), (3, 6)]
    x = 2
    solu = 3

    p = interpolate_natural_spline(points)
    y = p(x)

    print(f'points: {points}')
    print(f's({x}) = {y} <= {y == solu} => {solu}')
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

        p = interpolate_natural_spline(points)
        polynoms.append(p)

    x = np.arange(-5, 5, 0.01)
    plt.plot(x, func(x))
    legend = ['$f(x)$']
    for m, p in zip(ms, polynoms):
        plt.plot(x, [p(val) for val in x])
        legend.append('$s_{' + str(m) + '}(x)$')
    plt.legend(legend)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'Natural Cubic Splines for $f(x) = \frac{1}{1 + x^2}$')
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

        p = interpolate_natural_spline(points)
        polynoms.append(p)

    x = np.arange(-5, 5, 0.01)
    plt.plot(x, func(x))
    legend = ['$f(x)$']
    for m, p in zip(ms, polynoms):
        plt.plot(x, [p(val) for val in x])
        legend.append('$s_{' + str(m) + '}(x)$')
    plt.legend(legend)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(r'Natural Cubic Splines for $f(x) = \frac{1}{1 + x^2}$')
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
        
        p_1 = interpolate_natural_spline(points_1)
        polynoms.append(p_1)
        p_2 = interpolate_natural_spline(points_2)
        polynoms.append(p_2)

    x = np.arange(0, 1, 0.01)
    plt.plot(func_1(x), func_2(x))
    legend = [r'$\gamma(t)$']
    for m, index in zip(ms, range(0, len(polynoms), 2)):
        plt.plot([polynoms[index](val) for val in x], [polynoms[index+1](val) for val in x])
        legend.append('$s_{' + str(m) + '}(t)$')
    plt.legend(legend)
    plt.xlabel('x')
    plt.ylabel('y')
    # \binom as workaround for missing vectors in matplotlib's mathtext
    plt.title(r'Natural Cubic Splines for $\gamma(t) = \frac{1}{1 + t} \binom{\cos(3 \pi t)}{\sin(3 \pi t)}$')
    plt.grid(True)
    plt.show()
    print('-' * 30)
