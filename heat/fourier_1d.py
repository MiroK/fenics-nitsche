from math import pi, sin, cos, sqrt, log
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    'Step function.'
    n = len(x)
    y = np.ones(n)
    y[:n/4] = 0
    y[3*n/4:] = 0
    return y


def a(k):
    'Cosine coefficients.'
    # Note that doubling the coef. and only using cos. is a good approx.
    return 2*sin(k*pi*0.25)*cos(k*pi/2)/k/pi


def b(k):
    'Sine coefficients.'
    # Note that doubling the coef. and only using sin. is a good approx.
    return 2*sin(k*pi*0.25)*sin(k*pi/2)/k/pi


def F(x, n, series='full'):
    'Fourier series of f.'
    if series == 'full':
        return F(x, n, 'sine') + F(x, n, 'cosine')

    elif series == 'cosine':
        result = 0.25
        for k in range(1, n):
            result += a(k)*np.cos(k*pi*x)

    elif series == 'sine':
        result = 0
        for k in range(1, n):
            result += b(k)*np.sin(k*pi*x)

    return result

x = np.linspace(0, 1, 100)
y = f(x)

# Compute Fourier series
series = 'full'
ns = [10, 20, 40, 80, 160, 320]
ys = []
es = []
for n in ns:
    # Fourier series with n elements
    ys.append(F(x, n=n, series=series))
    # Error
    e = abs(ys[-1] - y)**2
    es.append(sqrt(np.trapz(e, x)))

# Plot
plt.figure()
plt.plot(x, y, label='$f$')

for i, (n, y, e) in enumerate(zip(ns, ys, es)):
    plt.plot(x, y, label='$F(f)_{%d},\,e=%.2E$' % (n, e))
    if i > 0:
        print -log(e/es[i-1])/log(n/ns[i-1])

plt.legend(loc='best')
plt.show()
