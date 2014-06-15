from sympy import *

x, y = symbols('x y')

laplace = lambda u: diff(u, x, 2) + diff(u, y, 2)

grad = lambda u: [diff(u, x, 1), diff(u, y, 1)]

u = x*y + sin(pi*x)*cos(2*pi*y)

grad_u = grad(u)

# top n=(0, 1) y==1
n = [0, 1]
print -grad_u[0]*n[0] - grad_u[1]*n[1]

# bottom n=(0, -1) y==0
n = [0, -1]
print -grad_u[0]*n[0] - grad_u[1]*n[1]

# left n=(-1, 0) x==0
n = [-1, 0]
print -grad_u[0]*n[0] - grad_u[1]*n[1]

# right n=(1, 0) x==1
n = [1, 0]
print -grad_u[0]*n[0] - grad_u[1]*n[1]
