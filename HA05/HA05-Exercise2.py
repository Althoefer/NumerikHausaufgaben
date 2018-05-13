import numpy as np


def zerlegung_mit_pivot(A):
    A = np.copy(A)
    x = np.array([i for i in range(len(A))], dtype=np.int)
    for i in range(len(A)):
        # find max entry in row and exchange rows
        x[i] = i + np.argmax(abs(A[i:, i]))
        A[i, :], A[x[i], :] = A[x[i], :], np.copy(A[i, :])
        for j in range(i + 1, len(A)):
            A[j, i] = A[j, i] / A[i, i]
            A[j, i + 1:] = -1 * A[j, i] * A[i, i + 1:] + A[j, i + 1:]
    return A, x


def permutation(p, x):
    x = np.copy(x)
    for row, row_to_swap_with in enumerate(p):
        x[row], x[row_to_swap_with] = x[row_to_swap_with], np.copy(x[row])
    return x


def vorwaerts(LU, x):
    x = np.copy(x)
    for row in range(len(LU)):
        x[row + 1:] = x[row + 1:] - x[row] * LU[row + 1:, row]
    return x


def rueckwaerts(LU, x):
    x = np.copy(x)
    for row in reversed(range(len(LU))):
        x[row] = x[row] / LU[row, row]
        x[:row] = x[:row] - x[row] * LU[:row, row]
    return x


def nachiteration(A, b, LU, p, x, epsilon = 10 ** -8):
    x = np.copy(x)
    
    r = b - np.dot(A, x)
    temp = permutation(p, r)
    temp = vorwaerts(LU, temp)
    p_k = rueckwaerts(LU, temp)
    x = x + p_k
    while np.linalg.norm(p_k) / np.linalg.norm(x) >= epsilon:
        r = b - np.dot(A, x)
        temp = permutation(p, r)
        temp = vorwaerts(LU, temp)
        p_k = rueckwaerts(LU, temp)
        x = x + p_k
    return x

if __name__ == '__main__':
    print('-' * 30)
    print('LU mit Pivot + Nachiteration')
    print('-' * 30)

    print('Exercise A')
    print('-' * 30)
    
    dims = [40, 50, 60]
    for n in dims:
        print(f'dimension: {n}')

        A = np.eye(n, dtype=np.float64)
        A[:, -1] = np.array([1 for _ in range(n)], dtype=np.float64)
        for i in range(n):
            A[i + 1:, i] = -1

        # zerlegungen
        LU, p = zerlegung_mit_pivot(A)

        b = np.array([3 - i for i in range(1, n)] + [2 - n], dtype=np.float64)
        print(f'input: {b}')

        # solve LGS
        y = permutation(p, b)
        y = vorwaerts(LU, y)
        x = rueckwaerts(LU, y)
        x = nachiteration(A, b, LU, p, x)
        print(f'solution LU w. pivot and iterative refinement: {x}')
        print('-' * 30)

    print('Exercise B')
    print('-' * 30)

    dims = [i for i in range(10, 16)]
    for n in dims:
        print(f'dimension: {n}')

        A = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if j > i:
                    A[i, j] = i + j + 2

        # zerlegungen
        LU, p = zerlegung_mit_pivot(A)

        b = np.array([1] + [0 for _ in range(n - 1)], dtype=np.float64)
        print(f'input: {b}')

        # solve LGS
        y = permutation(p, b)
        y = vorwaerts(LU, y)
        x = rueckwaerts(LU, y)
        x = nachiteration(A, b, LU, p, x)
        print(f'solution LU w. pivot and iterative refinement: {x}')
        print('-' * 30)
