'''
    Testing Poisson problem implementations from Freund, Sternberg
    'On weakly imposed boundary conditions for second order problem', 1995

    Convergence study on unit circle.
    Mixed bcs are considered; u = u_exact on \Gamma_D
                              grad(u).n = flux_exact on \Gamma_N
'''

from collections import namedtuple
import matplotlib.pyplot as plt
from math import log as ln
from dolfin import *
import numpy as np

set_log_level(WARNING)

Result = namedtuple('Result', ['h', 'L2', 'H10', 'H1'])

_ = '2*pi*(2*pi*x[0]*x[0]*sin(pi*(x[0]*x[0] + x[1]*x[1]))*cos(pi*(x[0] - x[1]))\
    + 2*pi*x[0]*sin(pi*(x[0] - x[1]))*cos(pi*(x[0]*x[0] + x[1]*x[1]))\
    + 2*pi*x[1]*x[1]*sin(pi*(x[0]*x[0] + x[1]*x[1]))*cos(pi*(x[0] - x[1]))\
    - 2*pi*x[1]*sin(pi*(x[0] - x[1]))*cos(pi*(x[0]*x[0] + x[1]*x[1]))\
    + pi*sin(pi*(x[0]*x[0] + x[1]*x[1]))*cos(pi*(x[0] - x[1]))\
    - 2*cos(pi*(x[0] - x[1]))*cos(pi*(x[0]*x[0] + x[1]*x[1])))'
f = Expression(_)

_ = 'x[0]*(2*pi*x[0]*cos(pi*(x[0] - x[1]))*cos(pi*(x[0]*x[0] + x[1]*x[1]))\
    - pi*sin(pi*(x[0] - x[1]))*sin(pi*(x[0]*x[0] + x[1]*x[1])))\
    + x[1]*(2*pi*x[1]*cos(pi*(x[0] - x[1]))*cos(pi*(x[0]*x[0] + x[1]*x[1]))\
    + pi*sin(pi*(x[0] - x[1]))*sin(pi*(x[0]*x[0] + x[1]*x[1])))'
flux_exact = Expression(_)
u_exact = Expression('sin(pi*(x[0]*x[0] + x[1]*x[1]))*cos(pi*(x[0] - x[1]))')


def gamma_D(x, on_boundary):
    return on_boundary and x[1] < DOLFIN_EPS


def gamma_N(x, on_boundary):
    return on_boundary and x[1] > - DOLFIN_EPS

Gamma_D = AutoSubDomain(gamma_D)
Gamma_N = AutoSubDomain(gamma_N)


def strong_poisson(N):
    'Standard formulation with strongly imposed bcs.'
    mesh = CircleMesh(Point(0., 0.), 1, 1./N)

    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    boundaries = FacetFunction('size_t', mesh, 0)
    ds = Measure('ds')[boundaries]
    Gamma_D.mark(boundaries, 1)
    Gamma_N.mark(boundaries, 2)
    dirichlet_bdry = 1
    neumann_bdry = 2

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx + inner(flux_exact, v)*ds(neumann_bdry)
    bc = DirichletBC(V, u_exact, boundaries, dirichlet_bdry)

    uh = Function(V)
    solve(a == L, uh, bc)

    # plot(uh, title='numeric')
    # plot(u_exact, mesh=mesh, title='exact')
    # interactive()

    # Compute norm of error
    E = FunctionSpace(mesh, 'DG', 4)
    uh = interpolate(uh, E)
    u = interpolate(u_exact, E)
    e = uh - u

    norm_L2 = assemble(inner(e, e)*dx, mesh=mesh)
    norm_H10 = assemble(inner(grad(e), grad(e))*dx, mesh=mesh)
    norm_H1 = norm_L2 + norm_H10

    norm_L2 = sqrt(norm_L2)
    norm_H1 = sqrt(norm_H1)
    norm_H10 = sqrt(norm_H10)

    # Check whether Neumann holds
    # x = np.linspace(0, 1, 100)
    # y = np.ones(len(x))
    # gradu_dot_n = project(grad(u)[1], V)
    # z = np.asarray([gradu_dot_n(p, q) for p, q in zip(x, y)])
    # Z = x + pi*np.sin(pi*x)*np.cos(pi)
    # plt.figure()
    # plt.plot(x, z, label='numeric')
    # plt.plot(x, Z, label='exact')
    # plt.legend()
    # plt.show()

    return Result(h=mesh.hmin(), L2=norm_L2, H1=norm_H1, H10=norm_H10)


def nitsche1_poisson(N):
    'Classical (symmetric) Nitsche formulation.'
    mesh = CircleMesh(Point(0., 0.), 1, 1./N)

    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    beta_value = 10
    beta = Constant(beta_value)

    h_E = mesh.ufl_cell().max_facet_edge_length
    n = FacetNormal(mesh)

    boundaries = FacetFunction('size_t', mesh, 0)
    ds = Measure('ds')[boundaries]
    Gamma_D.mark(boundaries, 1)
    Gamma_N.mark(boundaries, 2)
    neumann_bdry = 2
    dirichlet_bdry = 1

    # Include Dirichlet
    a = inner(grad(u), grad(v))*dx\
        - inner(dot(grad(u), n), v)*ds(dirichlet_bdry)\
        - inner(u, dot(grad(v), n))*ds(dirichlet_bdry)\
        + beta*h_E**-1*inner(u, v)*ds(dirichlet_bdry)

    L = inner(f, v)*dx + inner(flux_exact, v)*ds(neumann_bdry)\
        - inner(u_exact, dot(grad(v), n))*ds(dirichlet_bdry)\
        + beta*h_E**-1*inner(u_exact, v)*ds(dirichlet_bdry)

    uh = Function(V)
    solve(a == L, uh)

    # plot(uh, title='numeric')
    # plot(u_exact, mesh=mesh, title='exact')
    # interactive()

    # Compute norm of error
    E = FunctionSpace(mesh, 'DG', 4)
    uh = interpolate(uh, E)
    u = interpolate(u_exact, E)
    e = uh - u

    norm_L2 = assemble(inner(e, e)*dx, mesh=mesh)
    norm_H10 = assemble(inner(grad(e), grad(e))*dx, mesh=mesh)
    norm_edge = assemble(h_E**-1*inner(e, e)*ds(dirichlet_bdry))

    norm_H1 = norm_L2 + norm_H10 + norm_edge
    norm_L2 = sqrt(norm_L2)
    norm_H1 = sqrt(norm_H1)
    norm_H10 = sqrt(norm_H10)

    return Result(h=mesh.hmin(), L2=norm_L2, H1=norm_H1, H10=norm_H10)


def nitsche2_poisson(N):
    'Unsymmetric Nitsche formulation.'
    mesh = CircleMesh(Point(0., 0.), 1, 1./N)

    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    beta_value = 10
    beta = Constant(beta_value)
    h_E = mesh.ufl_cell().max_facet_edge_length
    n = FacetNormal(mesh)

    boundaries = FacetFunction('size_t', mesh, 0)
    ds = Measure('ds')[boundaries]
    Gamma_D.mark(boundaries, 1)
    Gamma_N.mark(boundaries, 2)
    neumann_bdry = 2
    dirichlet_bdry = 1

    a = inner(grad(u), grad(v))*dx\
        - inner(dot(grad(u), n), v)*ds(dirichlet_bdry)\
        + inner(u, dot(grad(v), n))*ds(dirichlet_bdry)\
        + beta*h_E**-1*inner(u, v)*ds(dirichlet_bdry)
    # The symmetry is lost but beta is not tied anymore to constant from
    # inverse estimate

    L = inner(f, v)*dx + inner(flux_exact, v)*ds(neumann_bdry)\
        + inner(u_exact, dot(grad(v), n))*ds(dirichlet_bdry)\
        + beta*h_E**-1*inner(u_exact, v)*ds(dirichlet_bdry)

    uh = Function(V)
    solve(a == L, uh)

    # plot(uh, title='numeric')
    # plot(u_exact, mesh=mesh, title='exact')
    # interactive()

    # Compute norm of error
    E = FunctionSpace(mesh, 'DG', 4)
    uh = interpolate(uh, E)
    u = interpolate(u_exact, E)
    e = uh - u

    norm_L2 = assemble(inner(e, e)*dx, mesh=mesh)
    norm_H10 = assemble(inner(grad(e), grad(e))*dx, mesh=mesh)
    norm_edge = assemble(h_E**-1*inner(e, e)*ds(dirichlet_bdry))
    norm_H1 = norm_L2 + norm_H10 + norm_edge

    norm_L2 = sqrt(norm_L2)
    norm_H1 = sqrt(norm_H1)
    norm_H10 = sqrt(norm_H10)

    return Result(h=mesh.hmin(), L2=norm_L2, H1=norm_H1, H10=norm_H10)

# -----------------------------------------------------------------------------

methods = [strong_poisson, nitsche1_poisson, nitsche2_poisson]
method = methods[2]
norm_type = 'H1'

R = method(N=4)
h_ = R.h
e_ = getattr(R, norm_type)
for N in [8, 16, 32, 64, 128]:
    R = method(N)
    h = R.h
    e = getattr(R, norm_type)
    rate = ln(e/e_)/ln(h/h_)
    print '{h:.3E} {e:.3E} {rate:.2f}'.format(h=h, e=e, rate=rate)
