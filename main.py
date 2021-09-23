import math
import random
import matplotlib.pyplot as plt
import numpy as np

from scipy import integrate


def func1(x):
    return math.exp(-x ** 4) * math.sqrt(1 + (x ** 4))


def func2(x, y):
    return 1 / ((x ** 2) + (y ** 4))


def montecarlo(f, a, b, n):
    res = sum([f(random.uniform(a, b)) for _ in range(n)])
    res = (b - a) * res / n
    return res


def montecarlo_geom(f, xa, xb, ya, yb, n):
    ys = list()
    fs = list()

    i = xa
    while i <= xb:
        ys.append(ya(i))
        ys.append(yb(i))

        i += 0.001

    ymin = min(ys)
    ymax = max(ys)

    i = xa
    while i <= xb:
        j = ymin
        while j <= ymax:
            fs.append(f(i, j))
            j += 0.001

        i += 0.001

    fmin = min(fs)
    fmax = max(fs)

    k = 0
    for _ in range(n):
        x = random.uniform(xa, xb)
        y = random.uniform(ymin, ymax)
        f_ = random.uniform(fmin, fmax)

        if ya(x) <= y <= yb(x) and f_ <= f(x, y):
            k += 1

    res = abs(xa - xb) * (ymax - ymin) * (fmax - fmin) * k / n
    return res


a1 = -6
b1 = 6

xa2 = 1
xb2 = 2
ya2 = lambda x: -math.sqrt(4 - (x ** 2))
yb2 = lambda x: math.sqrt(4 - (x ** 2))

true_integral1 = 2.00005

yvals = list()
xvals = list()
for n_ in range(10 ** 5, 10 ** 6, 10 ** 5):
    val = montecarlo(func1, a1, b1, n_)
    xvals.append(n_)
    yvals.append(abs(true_integral1 - val))

plt.plot(xvals, yvals)
plt.ylabel('Разница')
plt.xlabel('Число точек')
plt.title('Первый интеграл')
plt.show()

true_integral2 = integrate.dblquad(func2, xa2, xb2, ya2, yb2)[0]
# 0.65596

yvals = list()
xvals = list()
for n_ in range(10 ** 5, 10 ** 6, 10 ** 5):
    val = montecarlo_geom(func2, xa2, xb2, ya2, yb2, n_)
    print(val)
    xvals.append(n_)
    yvals.append(abs(true_integral2 - val))

plt.plot(xvals, yvals)
plt.ylabel('Разница')
plt.xlabel('Число точек')
plt.title('Второй интеграл')
plt.show()
