'''
    Solve the heat equation
    [Ryan C. Daileda, Two dimensional heat equation]

        T_t = kappa(T_xx + T_yy) in [0, 1]^2

    The temperature is prescribed on each edge in a form of a step function:

        T(x=0, y, t) = g0(y) = 0
        T(x=1, y, t) = g2(y) = 75y if 0 < y <= 2/3 else 150(1-y)
        T(x, y=0, t) = g1(x) = 0
        T(x, y=1, t) = g3(x) = 0

    Initial condition

        T(x, y, 0) = 0

    Compare solution obtained with classical FEM formulation, Nitsche FEM
    formulation and an analytical solution. The solution are compared at
    equilibrium state.
'''

from math import exp, sin, pi, sinh
from math import log as ln
from dolfin import *
import numpy as np

# Problem parameters
kappa = 1
N = 200  # Number of terms in Fourier representation of the solution
time_step = 1e-3  # Smaller then h**2/kappa for all meshes. Still, C-N
                  # should be uncoditionally stable
T_final = 0.5  # Time by which the temperature should reach equilibrium


def c(k):
    'Fourier coefficients for analytical equilibrium solution.'
    return 450*sin(2*k*pi/3)/(k*pi)**2/sinh(k*pi)


def T_e(x, y):
    'Analytical equilibrium solution'
    assert x.shape == y.shape

    result = np.zeros(x.shape)
    for k in range(1, N):
        result += c(k)*np.sin(k*pi*y)*np.sinh(k*pi*x)
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
        # Assign computed temperature
        self.f.vector()[:] = T_e(x, y)

    def __call__(self):
        'Add the time dependent part and return complete solution.'
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
    def eval(self, values, x):
        if between(x[1], (0, 2./3.)):
            values[0] = 75*x[1]
        else:
            values[0] = 150*(1-x[1])


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
    bc_l = DirichletBC(V, Constant(0.), left)
    bc_b = DirichletBC(V, Constant(0.), bottom)
    bc_r = DirichletBC(V, BoundarySource(), right)
    bc_t = DirichletBC(V, Constant(0.), top)
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
    g1 = Constant(0.)
    g2 = Constant(0.)
    g3 = BoundarySource()
    g4 = Constant(0.)

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

print 'mesh classic nitsche'
for i, N in enumerate([8, 16, 32, 64]):
    # Get the analytical solution in higher order space
    mesh = UnitSquareMesh(N, N)
    h = mesh.hmin()
    E = FunctionSpace(mesh, 'CG', 4)
    u = AnalyticalSolution(E)()

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
        print '%.2E    %.2f     %.2f' % (hs[-1], rate_c, rate_n)
