'''
    Solve the heat equation

        T_t = kappa(T_xx + T_yy) in [0, 1]^2

    The temperature is prescribed on each edge in a form of a step function:

        T(x=0, y, t) = g0(y) = T0/eps if abs(y - 0.5) < eps else 0
        T(x=1, y, t) = g2(y) = T2/eps if abs(y - 0.5) < eps else 0
        T(x, y=0, t) = g1(x) = T1/eps if abs(x - 0.5) < eps else 0
        T(x, y=1, t) = g3(x) = T3/eps if abs(x - 0.5) < eps else 0

    Initial condition

        T(x, y, 0) = 0

    Compare solution obtained with classical FEM formulation, Nitsche FEM
    formulation and an analytical solution. The solution are compared at
    equilibrium state.
'''

from math import exp, sin, pi
from math import log as ln
from dolfin import *
import numpy as np

# Problem parameters
eps = 0.25
T0 = 1         # Note that T0/eps is the actual value!
T1 = 1
T2 = 1
T3 = 1
kappa = 1
N = 100  # Number of terms in Fourier representation of the solution
time_step = 1e-3  # Smaller then h**2/kappa for all meshes. Still, C-N
                  # should be uncoditionally stable
T_final = 0.5  # Time by which the temperature should reach equilibrium

# Compute the analytical solution:
# The solution is sought as T = T` + T_e, where T_e solves
#
# T_e_xx + T_e_yy = 0, + bcs.
#
# Further T_e is a linear combination of T_i_e, the solutions to Laplace
# problem + # homog bcs everywhere but on i-th edge.
# The code is not refactored to comply with notes

# The equlibrium part, T_e
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


# The deviatoric part, T`
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


class AnalyticalSolution(object):
    'Analytical solution to heat problem as function in V'
    def __init__(self, V):
        'Precompute the equilibrium part of the solution.'
        self.f = Function(V)

        dofmap = V.dofmap()
        mesh = V.mesh()
        dof_x = dofmap.tabulate_all_coordinates(mesh).reshape((-1, 2))
        x, y = dof_x[:, 0], dof_x[:, 1]
        T_e = T_0_e(x, y) + T_1_e(x, y) + T_2_e(x, y) + T_3_e(x, y)

        self.x, self.y, self.T_e = x, y, T_e

    def __call__(self, t):
        'Add the time dependent part and return complete solution.'
        T_p = T_prime(self.x, self.y, t)
        T = self.T_e + T_p
        self.f.vector()[:] = T
        return self.f


# Get the solution using FEM formulations
# Define boundaries
def left(x, on_boundary):
    return on_boundary and near(x[0], 0)


def right(x, on_boundary):
    return on_boundary and near(x[0], 1)


def bottom(x, on_boundary):
    return on_boundary and near(x[1], 0)


def top(x, on_boundary):
    return on_boundary and near(x[1], 1)


# Define heating on the boundaries
class BoundarySource(Expression):
    def __init__(self, magnitude, i):
        self.mag = magnitude
        self.i = i

    def eval(self, values, x):
        if abs(x[self.i] - 0.5) < eps + DOLFIN_EPS:
            values[0] = self.mag
        else:
            values[0] = 0


def standard(N):
    'Heat problem solution using classical formulation.'

    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, 'CG', 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    k = Constant(kappa)
    dt = Constant(time_step)

    u0 = Function(V)           # Initial/previous solution
    u_cn = 0.5*(u + u0)        # Crank-Nicolson discretization
    F = dt**-1*inner(u - u0, v)*dx + inner(k*grad(u_cn), grad(v))*dx
    a, L = system(F)

    # Boundary conditions
    bc_l = DirichletBC(V, BoundarySource(T0/eps, 1), left)
    bc_b = DirichletBC(V, BoundarySource(T1/eps, 0), bottom)
    bc_r = DirichletBC(V, BoundarySource(T2/eps, 1), right)
    bc_t = DirichletBC(V, BoundarySource(T3/eps, 0), top)
    bcs = [bc_l, bc_b, bc_r, bc_t]

    # LSH matrix is not time dependent
    A = assemble(a)

    t = 0
    uh = Function(V)
    while t < T_final:
        t += dt(0)

        b = assemble(L)
        [bc.apply(A, b) for bc in bcs]

        solve(A, uh.vector(), b)
        u0.assign(uh)
        # plot(u0)

    return u0


def nitsche(N):
    'Heat problem using Nitsche formulation to set boundary conditions.'

    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, 'CG', 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    k = Constant(kappa)
    dt = Constant(time_step)

    # Mark the boundaries and associate surface measure with them
    boundaries = FacetFunction('size_t', mesh, 0)
    AutoSubDomain(left).mark(boundaries, 1)
    AutoSubDomain(bottom).mark(boundaries, 2)
    AutoSubDomain(right).mark(boundaries, 3)
    AutoSubDomain(top).mark(boundaries, 4)
    ds = Measure('ds')[boundaries]

    # Define Nitsche variables
    gamma = Constant(4)
    h_E = mesh.ufl_cell().max_facet_edge_length
    n = FacetNormal(mesh)

    # Create boundary functions
    g1 = BoundarySource(T0/eps, 1)
    g2 = BoundarySource(T1/eps, 0)
    g3 = BoundarySource(T2/eps, 1)
    g4 = BoundarySource(T3/eps, 0)

    u0 = Function(V)
    u_cn = 0.5*(u + u0)
    F = dt**-1*inner(u - u0, v)*dx + inner(k*grad(u_cn), grad(v))*dx\
        - inner(dot(k*grad(u_cn), n), v)*ds('everywhere')\
        - inner(dot(k*grad(v), n), u_cn)*ds('everywhere')\
        + gamma/h_E*inner(u_cn, v)*ds('everywhere')\
        + inner(dot(k*grad(v), n), g1)*ds(1)\
        - gamma/h_E*inner(g1, v)*ds(1)\
        + inner(dot(k*grad(v), n), g2)*ds(2)\
        - gamma/h_E*inner(g2, v)*ds(2)\
        + inner(dot(k*grad(v), n), g3)*ds(3)\
        - gamma/h_E*inner(g3, v)*ds(3)\
        + inner(dot(k*grad(v), n), g4)*ds(4)\
        - gamma/h_E*inner(g4, v)*ds(4)\

    a, L = system(F)

    A = assemble(a)

    t = 0
    uh = Function(V)
    while t < T_final:
        t += dt(0)

        b = assemble(L)

        solve(A, uh.vector(), b)
        u0.assign(uh)
        # plot(u0)

    return u0

# ----------------------------------------------------------------------------

# Compute convergence rate
errors_c = []
errors_n = []
hs = []
for i, N in enumerate([4, 8, 16, 32, 64]):
    # Get the analytical solution in higher order space
    mesh = UnitSquareMesh(N, N)
    h = mesh.hmin()
    E = FunctionSpace(mesh, 'CG', 4)
    u = AnalyticalSolution(E)(t=T_final)

    # Get error from classical solution
    uh = standard(N)
    uh = interpolate(uh, E)
    e = u - uh
    error_c = sqrt(assemble(inner(e, e)*dx))

    # Get error from Nitsche solution
    uh = nitsche(N)
    uh = interpolate(uh, E)
    e = u - uh
    error_n = sqrt(assemble(inner(e, e)*dx))

    hs.append(h)
    errors_c.append(error_c)
    errors_n.append(error_n)

    if i > 0:
        rate_c = ln(errors_c[i]/errors_c[i-1])/ln(hs[i]/hs[i-1])
        rate_n = ln(errors_n[i]/errors_n[i-1])/ln(hs[i]/hs[i-1])
        print '%g %.2f %.2f' % (hs[-1], rate_c, rate_n)
