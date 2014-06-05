'''
    Using Dirichlet g, Neumann h and Robin boudnary conditions r with
    Nitsche method.
'''

from collections import namedtuple
from math import log as ln
from dolfin import *

set_log_level(WARNING)

Result = namedtuple('Result', ['h', 'L2', 'H10', 'H1'])

# Define problem data
f = Expression('25*pi*pi*(x[1] + sin(2*pi*x[1]))*sin(5*pi*x[0]/2)/4 +\
               4*pi*pi*sin(5*pi*x[0]/2)*sin(2*pi*x[1])')
u_exact = Expression('sin(5*pi*x[0]/2)*(x[1] + sin(2*pi*x[1]))')
G = u_exact
H = Expression('(2*pi*cos(2*pi*x[1]) + 1)*sin(5*pi*x[0]/2)')
R = Expression('-(2*pi*cos(2*pi*x[1]) + 1)*sin(5*pi*x[0]/2) + sin(5*pi*x[0]/2)')

# Define boundaries
def gamma_d(x, on_boundary):
    'Dirichlet boundary.'
    return on_boundary and near(x[0]*(1 - x[0]), 0)


def gamma_n(x, on_boundary):
    'Neumann boundary.'
    return on_boundary and near(x[1], 0)


def gamma_r(x, on_boundary):
    'Robin boundary.'
    return on_boundary and near(x[1], 1)

Gamma_d = AutoSubDomain(gamma_d)
Gamma_n = AutoSubDomain(gamma_n)
Gamma_r = AutoSubDomain(gamma_r)


def classic(N):
    'Standard formulation with strongly imposed bcs.'
    mesh = Mesh(Rectangle(0, 0, 1, 1), N)

    # Label boundaries
    boundaries = FacetFunction('size_t', mesh, 0)
    d_bdr, n_bdr, r_bdr = 1, 2, 3
    Gamma_d.mark(boundaries, d_bdr)
    Gamma_n.mark(boundaries, n_bdr)
    Gamma_r.mark(boundaries, r_bdr)
    ds = Measure('ds')[boundaries]

    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx - inner(u, v)*ds(r_bdr)
    L = inner(f, v)*dx - inner(H, v)*ds(n_bdr) - inner(R, v)*ds(r_bdr)
    bc = DirichletBC(V, G, boundaries, d_bdr)

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


def nitsche(N):
    'Nitsche formulation.'
    mesh = Mesh(Rectangle(0, 0, 1, 1), N)

    # Label boundaries
    boundaries = FacetFunction('size_t', mesh, 0)
    d_bdr, n_bdr, r_bdr = 1, 2, 3
    Gamma_d.mark(boundaries, d_bdr)
    Gamma_n.mark(boundaries, n_bdr)
    Gamma_r.mark(boundaries, r_bdr)
    ds = Measure('ds')[boundaries]

    V = FunctionSpace(mesh, 'CG', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    n = FacetNormal(mesh)
    beta = Constant(10)
    h = mesh.ufl_cell().max_facet_edge_length

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    # Add Dirichlet bcs
    a += - inner(dot(grad(u), n), v)*ds(d_bdr)\
        - inner(dot(grad(v), n), u)*ds(d_bdr)\
        + beta/h*inner(u, v)*ds(d_bdr)

    L += - inner(dot(grad(v), n), G)*ds(d_bdr) + beta/h*inner(G, v)*ds(d_bdr)

    # Add Neumann bcs
    L += -inner(H, v)*ds(n_bdr)

    # Add Robin
    a += -inner(u, v)*ds(r_bdr)

    L += -inner(R, v)*ds(r_bdr)

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
    norm_H1 = norm_L2 + norm_H10

    norm_L2 = sqrt(norm_L2)
    norm_H1 = sqrt(norm_H1)
    norm_H10 = sqrt(norm_H10)

    return Result(h=mesh.hmin(), L2=norm_L2, H1=norm_H1, H10=norm_H10)
# -----------------------------------------------------------------------------

methods = [classic, nitsche]
method = methods[1]
norm_type = 'L2'

Res = method(N=4)
h_ = Res.h
e_ = getattr(Res, norm_type)
for N in [8, 16, 32, 64, 128]:
    Res = method(N)
    h = Res.h
    e = getattr(Res, norm_type)
    rate = ln(e/e_)/ln(h/h_)
    print '{h:.3E} {e:.3E} {rate:.2f}'.format(h=h, e=e, rate=rate)
