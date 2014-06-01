import matplotlib.animation as animation
import matplotlib.pyplot as plt
from math import exp, sin, pi
import numpy as np

# Solve heat equation
#
# T_t = kappa*laplace(T)
#
# in [0, 1]^2 with temperature prescribed
# as a jump of width eps on each side. Magnitude of jump
# is Ti/eps.
#           T3
#     -----[  ]-----
#     |            |
#     _            _
#  T0 _   kappa    _ T2
#     |            |
#     |            |
#     -----[  ]-----
#           T1

# Problem parameters
eps = 0.25
T0 = 1         # Note that T0/eps is the actual value!
T1 = 1
T2 = 1
T3 = 1
kappa = 1
N = 100  # Number of terms in Fourier representation of the solutions

# The solution is sought as T = T` + T_e, where T_e, the equilibrium
# temperature, solve laplace(T_e) = 0 + bcs. Further T_e is is a linear
# combination of T_i_e, where T_i_e solves laplace(T_i_e) = ) and
# T_i_e is 0 on the boundary except on \Gamma_i where it matches Ti.

# The code is not refactored to reflect math/my notes better


def c0(n):
    'Coefficients for t_0_e.'
    return 4*T0*sin(n*pi*eps)*sin(n*pi/2)/n/pi/eps


def T_0_e(x, y, n_terms=N):
    'Equilibrium temperature that matches boundary conditions T0.'
    assert x.shape == y.shape
    result = np.zeros(x.shape)
    for n in range(1, n_terms):
        result += c0(n)*np.sin(n*pi*y)*np.exp(-n*pi*x)
    return result


def c1(n):
    'Coefficients for t_1_e.'
    return 4*T1*sin(n*pi*eps)*sin(n*pi/2)/n/pi/eps


def T_1_e(x, y, n_terms=N):
    'Equilibrium temperature that matches boundary conditions T1.'
    assert x.shape == y.shape
    result = np.zeros(x.shape)
    for n in range(1, n_terms):
        result += c1(n)*np.sin(n*pi*x)*np.exp(-n*pi*y)
    return result


def c2(n):
    'Coefficients for t_2_e.'
    return 4*T2*sin(n*pi*eps)*sin(n*pi/2)/n/pi/eps


def T_2_e(x, y, n_terms=N):
    'Equilibrium temperature that matches boundary conditions T2.'
    assert x.shape == y.shape
    result = np.zeros(x.shape)
    for n in range(1, n_terms):
        result += c2(n)*np.sin(n*pi*y)*np.exp(n*pi*(x-1))
    return result


def c3(n):
    'Coefficients for t_3_e.'
    return 4*T3*sin(n*pi*eps)*sin(n*pi/2)/n/pi/eps


def T_3_e(x, y, n_terms=N):
    'Equilibrium temperature that matches boundary conditions T3.'
    assert x.shape == y.shape
    result = np.zeros(x.shape)
    for n in range(1, n_terms):
        result += c3(n)*np.sin(n*pi*x)*np.exp(n*pi*(y-1))
    return result


X = np.linspace(0, 1, 20)
Y = np.linspace(0, 1, 20)
x, y = np.meshgrid(X, Y)

# Plot the equilibrium solution
T_e = T_0_e(x, y) + T_1_e(x, y) + T_2_e(x, y) + T_3_e(x, y)

min, max = T_e.min(), T_e.max()
levels = np.linspace(min, max, 10)

fig = plt.figure()
ax = fig.gca()
contour = ax.contour(x, y, T_e, levels=levels)
plt.clabel(contour, inline=1, fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])


def c(k, l):
    'Coefficients for the homog. heat problem'
    result = 0
    result += -2*c0(l)*((-1)**(k+1)*exp(-l*pi) + 1)/k/pi/(1 + (l/k)**2)
    result += -2*c1(k)*((-1)**(l+1)*exp(-k*pi) + 1)/l/pi/(1 + (k/l)**2)
    result += -2*c2(l)*((-1)**(k+1) + exp(-l*pi))/k/pi/(1 + (l/k)**2)
    result += -2*c3(k)*((-1)**(l+1) + exp(-k*pi))/l/pi/(1 + (k/l)**2)
    return result


def T_prime(x, y, t, n_terms=N):
    'Solution of the heat problem with homog. boundary conditions.'
    result = np.zeros(x.shape)
    for m in range(1, n_terms):
        for n in range(1, n_terms):
            result += c(m, n)*np.sin(m*pi*x)*np.sin(n*pi*y)*\
                np.exp(-kappa*t*((m*pi)**2 + (n*pi)**2))
    return result

ts = np.arange(0, 5E-1, 1E-3)
z = np.zeros((len(ts), x.shape[0], x.shape[1]))
for i, t in enumerate(ts):
    z[i] = T_prime(x, y, t) + T_e
    print t, z[i].min(), z[i].max()


def update_contourf(i):
    ax.cla()
    im = ax.contour(x, y, z[i], levels=levels)
    plt.clabel(im, inline=1, fontsize=10)
    plt.title(str(ts[i]))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    return im

fig = plt.figure()
im = plt.contourf(x, y, z[0])
ax = fig.gca()
ani = animation.FuncAnimation(fig, update_contourf, np.arange(1, len(ts)))
plt.show()
