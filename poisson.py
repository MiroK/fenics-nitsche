'''
    Testing Poisson problem implementations from Freund, Sternberg
    'On weakly imposed boundary conditions for second order problem', 1995
'''

from collections import namedtuple
from math import log as ln
from dolfin import *

set_log_level(WARNING)

Result = namedtuple('Result', ['h', 'L2', 'H10', 'H1'])
f = Expression('2*pi*pi*sin(pi*x[0])*sin(pi*x[1])')
u_exact = Expression('x[0]*x[1] + sin(pi*x[0])*sin(pi*x[1])')


def strong_poisson(N):
    'Standard formulation with strongly imposed bcs.'
    mesh = UnitSquareMesh(N, N)

    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx
    bc = DirichletBC(V, u_exact, DomainBoundary())

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

    return Result(h=mesh.hmin(), L2=norm_L2, H1=norm_H1, H10=norm_H10)


def nitsche1_poisson(N):
    'Classical (symmetric) Nitsche formulation.'
    mesh = UnitSquareMesh(N, N)

    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    beta_value = 2
    beta = Constant(beta_value)
    h_E = mesh.ufl_cell().max_facet_edge_length
    n = FacetNormal(mesh)

    a = inner(grad(u), grad(v))*dx - inner(dot(grad(u), n), v)*ds -\
        inner(u, dot(grad(v), n))*ds + beta*h_E**-1*inner(u, v)*ds

    L = inner(f, v)*dx -\
        inner(u_exact, dot(grad(v), n))*ds + beta*h_E**-1*inner(u_exact, v)*ds

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
    norm_edge = assemble(h_E**-1*inner(e, e)*ds)

    norm_H1 = norm_L2 + norm_H10 + norm_edge
    norm_L2 = sqrt(norm_L2)
    norm_H1 = sqrt(norm_H1)
    norm_H10 = sqrt(norm_H10)

    return Result(h=mesh.hmin(), L2=norm_L2, H1=norm_H1, H10=norm_H10)


def nitsche2_poisson(N):
    'Unsymmetric Nitsche formulation.'
    mesh = UnitSquareMesh(N, N)

    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    beta_value = 2
    beta = Constant(beta_value)
    h_E = mesh.ufl_cell().max_facet_edge_length
    n = FacetNormal(mesh)

    a = inner(grad(u), grad(v))*dx - inner(dot(grad(u), n), v)*ds +\
        inner(u, dot(grad(v), n))*ds + beta*h_E**-1*inner(u, v)*ds

    L = inner(f, v)*dx +\
        inner(u_exact, dot(grad(v), n))*ds + beta*h_E**-1*inner(u_exact, v)*ds

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
    norm_edge = assemble(h_E**-1*inner(e, e)*ds)
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
