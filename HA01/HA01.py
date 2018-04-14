# -*- coding: utf-8 -*-

# Exercise 1
from math import sqrt

def null1(p, q):
    x1 = -1 * p - sqrt(p ** 2 + q)
    x2 = -1 * p + sqrt(p ** 2 + q)
    return x1, x2

def null2(p, q):
    x1 = -1 * p - sqrt(p ** 2 + q)
    x2 = -1 * q / x1
    return x1, x2

if __name__ == "__main__":
    for exponent in [2, 4, 6, 7, 8]:
        p = 10 ** exponent
        q = 1
        print("Calculating for p = {} and q = {}".format(p, q))
        print(null1(p, q))
        print(null2(p, q))

# Exercise 2
from math import factorial, e

def exp1(n, x):
    s = 0.
    for k in range(n + 1):
        s += (x ** k / factorial(k))
    return s

def exp2(n, x):
    s = 0.
    for k in range(n + 1):
        s += (x ** (n - k) / factorial(n - k))
    return s

if __name__ == "__main__":
    for n in [5, 10, 20, 30]:
        for x in [0, 1, 2, 3, 10, 50]:
            print("Calculating for n = {} and x = {}".format(n, x))
            print("Exact Solution e^{} = {}".format(x, e ** x))
            print(exp1(n, x))
            print(exp2(n, x))

# Exercise 3
from math import log

def integ1(f, a, b, n):
    h = (b - a) / n
    s = 0.
    for i in range(n):
        s += f(a + i * h)
    return h * s

def integ2(f, a, b, n):
    h = (b - a) / n
    s = 0.
    for i in range(1, n):
        s += f(a + i * h)
    return (h / 2) * (f(a) + 2 * s + f(b))

def func1(x):
    return 1 / (x ** 2)

def func2(x):
    return log(x)

if __name__ == "__main__":
    print("Testing f(x) = 1 / (x ** 2)")
    for n in [5, 25, 100, 100000, 10000000]:
        print("Running with n = {}".format(n))
        print(integ1(func1, 0.1, 10, n))
        print(integ2(func1, 0.1, 10, n))
    print("Testing f(x) = ln(x)")
    for n in [5, 25, 100, 100000, 10000000]:
        print("Running with n = {}".format(n))
        print(integ1(func2, 1, 2, n))
        print(integ2(func2, 1, 2, n))

# Exercise 4
def integ_vec(f, a, b, n):
    h = (b - a) / n
    pos = [a + i * h for i in range(n)]
    pos = map(f, pos)
    return h * sum(pos)

def func1(x):
    return 1 / (x ** 2)

def func2(x):
    return log(x)

if __name__ == "__main__":
    print("Testing f(x) = 1 / (x ** 2)")
    for n in [5, 25, 100, 100000, 10000000]:
        print("Running with n = {}".format(n))
        print(integ_vec(func1, 0.1, 10, n))
    print("Testing f(x) = ln(x)")
    for n in [5, 25, 100, 100000, 10000000]:
        print("Running with n = {}".format(n))
        print(integ_vec(func2, 1, 2, n))

# Exercise 5
# from math import sqrt # already imported above

def sqrt1(k, x, x0):
    y = a = sqrt(x0) # pretend like we know sqrt(x0)
    for i in range(1, k + 1):
        a = (3 / (2 * i) - 1) * (x / x0 - 1) * a
        y = y + a
    return y

def sqrt2(k, x, x0):
    y = sqrt(x0) # pretend like we know sqrt(x0)
    for i in range(1, k + 1):
        y = 0.5 * (y + x / y)
    return y

if __name__ == "__main__":
    k = 10
    x = 2
    print("Testing sqrt({}) = {}".format(x, sqrt(x)))
    for x0 in [1, 4]:
        appr = sqrt1(k, x, x0)
        print("Taylor: x0 = {}: {} ==> diff: {}".format(x0, appr, abs(appr - sqrt(x))))
        appr = sqrt2(k, x, x0)
        print("Heron: x0 = {}: {} ==> diff: {}".format(x0, appr, abs(appr - sqrt(x))))
