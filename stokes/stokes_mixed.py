'''
    Comparison of Stokes problem formulation.

    We consider:
        classical symmetric, indefinite formulation
        symmetric Nitsche formulation [see dolfin-adjoint demo]
        skew-symmetric Nitsche formulation
            [R. Becker;
            Mesh adaptation for Dirichlet flow control via Nitsche's method]
            [J. Benk et al.;
            The Nitsche method of the N-S eqs for immersed and moving bdaries]

    Convergence study on unit square. Conditions and pressure and velocity
    are prescribed.
'''

from collections import namedtuple
from math import log as ln
from dolfin import *

Result = namedtuple('Result', ['h', 'u', 'p'])

re = 2
f = Expression(('-pi*sin(pi*x[0]) + 1 + (pi*pi*sin(pi*x[1]))/re', '0'), re=re)
u_exact = Expression(('sin(pi*x[1])', '0'))
p_exact = Expression('x[0] + cos(pi*x[0])')


def gamma_u(x, on_boundary):
    'Domain where velocity is prescribed.'
    return on_boundary and near(x[1]*(1 - x[1]), 0)


def gamma_p(x, on_boundary):
    'Domain where pressure is prescribed.'
    return on_boundary and near(x[0]*(1 - x[0]), 0)

Gamma_u = AutoSubDomain(gamma_u)
Gamma_p = AutoSubDomain(gamma_p)


def standard_stokes(N, symmetric=True):
    'Standard symmetric/skew-symmetric formulation.'

    mesh = Mesh(Rectangle(0, 0, 1, 1), N)

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    M = MixedFunctionSpace([V, Q])

    u, p = TrialFunctions(M)
    v, q = TestFunctions(M)

    Re = Constant(re)

    boundaries = FacetFunction('size_t', mesh, 0)
    u_bdr = 1
    Gamma_u.mark(boundaries, u_bdr)
    p_bdr = 2
    Gamma_p.mark(boundaries, p_bdr)
    ds = Measure('ds')[boundaries]
    n = FacetNormal(mesh)

    if symmetric:
        a = Re**-1*inner(grad(u), grad(v))*dx - p*div(v)*dx - q*div(u)*dx
    else:
        a = Re**-1*inner(grad(u), grad(v))*dx - p*div(v)*dx + q*div(u)*dx

    L = -inner(p_exact, dot(v, n))*ds(p_bdr) + inner(f, v)*dx
    bc = DirichletBC(M.sub(0), u_exact, boundaries, u_bdr)

    uph = Function(M)
    A, b = assemble_system(a, L, bc, exterior_facet_domains=boundaries)
    solve(A, uph.vector(), b)

    uh, ph = uph.split(True)

    # plot(uh, title='u numeric')
    # plot(ph, title='p numeric')
    # plot(u_exact, title='u exact', mesh=mesh)
    # plot(p_exact, title='p exact', mesh=mesh)
    # interactive()
    # Compute norm of error

    Eu = VectorFunctionSpace(mesh, 'DG', 5)
    Ep = FunctionSpace(mesh, 'CG', 4)

    uh = interpolate(uh, Eu)
    ph = interpolate(ph, Ep)
    u = interpolate(u_exact, Eu)
    p = interpolate(p_exact, Ep)
    eu = uh - u
    ep = ph - p

    error_u_L2 = assemble(eu**2*dx, mesh=mesh)
    error_u_H10 = assemble(grad(eu)**2*dx, mesh=mesh)
    error_u = sqrt(error_u_L2 + error_u_H10)

    error_p_L2 = assemble(ep**2*dx, mesh=mesh)
    error_p = sqrt(error_p_L2)

    return Result(h=mesh.hmin(), u=error_u, p=error_p)

def nitsche_stokes(N, symmetric=True):
    'Nitsche symmetric/skew-symmetric formulation.'

    mesh = Mesh(Rectangle(0, 0, 1, 1), N)

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    M = MixedFunctionSpace([V, Q])

    u, p = TrialFunctions(M)
    v, q = TestFunctions(M)

    Re = Constant(re)

    boundaries = FacetFunction('size_t', mesh, 0)
    u_bdr = 1
    Gamma_u.mark(boundaries, u_bdr)
    p_bdr = 2
    Gamma_p.mark(boundaries, p_bdr)
    ds = Measure('ds')[boundaries]
    n = FacetNormal(mesh)
    h_E = mesh.ufl_cell().max_facet_edge_length

    if symmetric:
        # Symmetric Nitsche formulation
        beta = Constant(10)
        a = Re**-1*inner(grad(u), grad(v))*dx - p*div(v)*dx - q*div(u)*dx\
            - Re**-1*inner(dot(grad(u), n), v)*ds(u_bdr)\
            + inner(p*n, v)*ds(u_bdr)\
            - Re**-1*inner(dot(grad(v), n), u)*ds(u_bdr)\
            + inner(q*n, u)*ds(u_bdr)\
            + beta/h_E/Re*inner(u, v)*ds(u_bdr)

        L = -inner(p_exact, dot(v, n))*ds(p_bdr) + inner(f, v)*dx\
            - Re**-1*inner(dot(grad(v), n), u_exact)*ds(u_bdr)\
            + inner(q*n, u_exact)*ds(u_bdr)\
            + beta/h_E/Re*inner(u_exact, v)*ds(u_bdr)
    else:
        # Skew-symmetric Nitsche : note `+ q*div(u)`, `-inner(qn, v) vs
        # `+ inner(pn, v)`
        gamma_1 = Constant(10)
        gamma_2 = Constant(10)

        a = Re**-1*inner(grad(u), grad(v))*dx - p*div(v)*dx + q*div(u)*dx\
            - Re**-1*inner(dot(grad(u), n), v)*ds(u_bdr)\
            + inner(p*n, v)*ds(u_bdr)\
            - Re**-1*inner(u, dot(grad(v), n))*ds(u_bdr)\
            - inner(q*n, u)*ds(u_bdr)\
            + gamma_1/h_E/Re*inner(u, v)*ds(u_bdr)\
            + gamma_2/h_E*inner(dot(u, n), dot(v, n))*ds(u_bdr)  # No Re**-1 !

        L = -inner(p_exact, dot(v, n))*ds(u_bdr)(p_bdr) + inner(f, v)*dx\
            - Re**-1*inner(u_exact, dot(grad(v), n))*ds(u_bdr)\
            - inner(q*n, u_exact)*ds(u_bdr)\
            + gamma_1/h_E/Re*inner(u_exact, v)*ds(u_bdr)\
            + gamma_2/h_E*inner(dot(u_exact, n), dot(v, n))*ds(u_bdr)

    uph = Function(M)
    A, b = assemble_system(a, L, exterior_facet_domains=boundaries)
    solve(A, uph.vector(), b)

    uh, ph = uph.split(True)

    # plot(uh, title='u numeric')
    # plot(ph, title='p numeric')
    # plot(u_exact, title='u exact', mesh=mesh)
    # plot(p_exact, title='p exact', mesh=mesh)
    # interactive()
    # Compute norm of error

    Eu = VectorFunctionSpace(mesh, 'DG', 5)
    Ep = FunctionSpace(mesh, 'CG', 4)

    uh = interpolate(uh, Eu)
    ph = interpolate(ph, Ep)
    u = interpolate(u_exact, Eu)
    p = interpolate(p_exact, Ep)
    eu = uh - u
    ep = ph - p

    error_u_L2 = assemble(eu**2*dx, mesh=mesh)
    error_u_H10 = assemble(grad(eu)**2*dx, mesh=mesh)
    error_u = sqrt(error_u_L2 + error_u_H10)

    error_p_L2 = assemble(ep**2*dx, mesh=mesh)
    error_p = sqrt(error_p_L2)

    return Result(h=mesh.hmin(), u=error_u, p=error_p)

methods = [standard_stokes, nitsche_stokes]
method = methods[1]
norm_type = 'u'
symmetric = False

R = method(N=4, symmetric=symmetric)
h_ = R.h
e_ = getattr(R, norm_type)
for N in [8, 16, 32, 64]:
    R = method(N=N, symmetric=symmetric)
    h = R.h
    e = getattr(R, norm_type)
    rate = ln(e/e_)/ln(h/h_)
    print '{h:.3E} {e:.3E} {rate:.2f}'.format(h=h, e=e, rate=rate)
