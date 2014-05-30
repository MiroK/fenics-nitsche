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

    Convergence study on unit square, only dirichlet bcs are considered
'''

from collections import namedtuple
from math import log as ln
from dolfin import *

set_log_level(WARNING)

Result = namedtuple('Result', ['h', 'u', 'p'])

re = 0.01
f = Expression(('-pi*sin(pi*x[0]) + pi*pi*sin(pi*x[1])/re + 1',
                'pi*pi*sin(pi*x[0])/re'), re=re)
u_exact = Expression(('x[1] + sin(pi*x[1])', 'x[0] + sin(pi*x[0])'))
p_exact = Expression('x[0] + cos(pi*x[0])')


def standard_stokes(N, symmetric=True):
    'Standard symmetric/skew-symmetric formulation.'

    mesh = Mesh(Rectangle(0, 0, 1, 1), N)

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    M = MixedFunctionSpace([V, Q])

    u, p = TrialFunctions(M)
    v, q = TestFunctions(M)

    Re = Constant(re)

    if symmetric:
        a = Re**-1*inner(grad(u), grad(v))*dx\
            - inner(p, div(v))*dx - inner(q, div(u))*dx
    else:
        a = Re**-1*inner(grad(u), grad(v))*dx\
            - inner(p, div(v))*dx + inner(q, div(u))*dx

    L = inner(f, v)*dx

    bc = DirichletBC(M.sub(0), u_exact, DomainBoundary())

    A, b = assemble_system(a, L, bc)
    uph = Function(M)
    solve(A, uph.vector(), b)

    uh, ph = uph.split(True)

    # Pressure determined up to constant. Fix it to match the p_exact
    Ph = ph.vector()
    normalize(Ph)
    Ph += 0.5

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
    'Symmetric or ske-symmetric Nitsche formulation.'

    mesh = Mesh(Rectangle(0, 0, 1, 1), N)

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    M = MixedFunctionSpace([V, Q])

    u, p = TrialFunctions(M)
    v, q = TestFunctions(M)

    Re = Constant(re)
    h_E = mesh.ufl_cell().max_facet_edge_length
    n = FacetNormal(mesh)

    if symmetric:
        a = Re**-1*inner(grad(u), grad(v))*dx\
            - inner(p, div(v))*dx - inner(q, div(u))*dx\

        L = inner(f, v)*dx

        # Add terms forcing dirichlet bcs
        beta_value = 10
        beta = Constant(beta_value)

        a += - Re**-1*inner(dot(grad(u), n), v)*ds\
            + inner(p*n, v)*ds\
            - Re**-1*inner(dot(grad(v), n), u)*ds\
            + inner(q*n, u)*ds\
            + beta/h_E/Re*inner(u, v)*ds

        L += - Re**-1*inner(dot(grad(v), n), u_exact)*ds\
            + inner(q*n, u_exact)*ds\
            + beta/h_E/Re*inner(u_exact, v)*ds
    else:
        # Add terms forcing dirichlet bcs
        gamma_1_value = 4
        gamma_2_value = 8
        gamma_1 = Constant(gamma_1_value)
        gamma_2 = Constant(gamma_2_value)

        a = Re**-1*inner(grad(u), grad(v))*dx\
            - inner(p, div(v))*dx + inner(q, div(u))*dx

        L = inner(f, v)*dx

        # Add terms enforcing boundary conditions
        a += - Re**-1*inner(dot(grad(u), n), v)*ds\
            + inner(p*n, v)*ds\
            - Re**-1*inner(u, dot(grad(v), n))*ds\
            - inner(q*n, u)*ds\
            + gamma_1/h_E/Re*inner(u, v)*ds\
            + gamma_2/h_E*inner(dot(u, n), dot(v, n))*ds  # No Re**-1 !

        L += - Re**-1*inner(u_exact, dot(grad(v), n))*ds\
            - inner(q*n, u_exact)*ds\
            + gamma_1/h_E/Re*inner(u_exact, v)*ds\
            + gamma_2/h_E*inner(dot(u_exact, n), dot(v, n))*ds  # No Re**-1

    A, b = assemble_system(a, L)
    uph = Function(M)
    solve(A, uph.vector(), b)

    uh, ph = uph.split(True)

    # Pressure determined up to constant. Fix it to match the p_exact
    Ph = ph.vector()
    normalize(Ph)
    Ph += 0.5

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

# ----------------------------------------------------------------------------

methods = [standard_stokes, nitsche_stokes]
method = methods[1]
norm_type = 'u'
symmetric = True

R = method(N=4, symmetric=symmetric)
h_ = R.h
e_ = getattr(R, norm_type)
for N in [8, 16, 32, 64]:
    R = method(N=N, symmetric=symmetric)
    h = R.h
    e = getattr(R, norm_type)
    rate = ln(e/e_)/ln(h/h_)
    print '{h:.3E} {e:.3E} {rate:.2f}'.format(h=h, e=e, rate=rate)
