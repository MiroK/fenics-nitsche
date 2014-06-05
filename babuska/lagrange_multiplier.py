'''
Enforcing Dirichlet boundary conditions weakly using Lagrange multipler techni-
que introduced in
    [Babuska; The finite element method with lagrangian multipliers]

    Simple Poisson problem:
        -laplace(u) = f in [0, 1]^2
                  u = g on boundary
'''

from collections import namedtuple
from dolfin import *

Result = namedtuple('Result', ['h', 'H1', 'L2'])

u_exact = Expression('x[0]*x[1] + sin(pi*x[0])*cos(2*pi*x[1])')
f = Expression('5*pi*pi*sin(pi*x[0])*cos(2*pi*x[1])')


def lagrange_solver(N):
    'Solve the Poisson problem.'

    mesh = UnitSquareMesh(N, N)

    V = FunctionSpace(mesh, 'CG', 1)
    W = MixedFunctionSpace([V, V])

    u, lambda_ = TrialFunctions(W)
    v, mu_ = TestFunctions(W)

    a = inner(grad(u), grad(v))*dx + inner(lambda_, v)*ds + inner(mu_, u)*ds
    L = inner(f, v)*dx + inner(mu_, u_exact)*ds

    A, b = assemble_system(a, L)
    # Eliminate dofs of W.sub(1) which are unset since the form has ds, not dx.
    A.ident_zeros()

    Uh = Function(W)
    solve(A, Uh.vector(), b)

    uh, lambda_h = Uh.split(True)
    # plot(uh, interactive=True)

    error_L2 = errornorm(u_exact, uh, 'L2')
    error_H1 = errornorm(u_exact, uh, 'H1')

    return Result(h=mesh.hmin(), H1=error_H1, L2=error_L2)

# -----------------------------------------------------------------------------

norm_type = 'L2'
R = lagrange_solver(N=4)
h_ = R.h
e_ = getattr(R, norm_type)
for N in [8, 16, 32, 64, 128]:
    R = lagrange_solver(N=N)
    h = R.h
    e = getattr(R, norm_type)
    rate = ln(e/e_)/ln(h/h_)
    print '{h:.3E} {e:.3E} {rate:.2f}'.format(h=h, e=e, rate=rate)
