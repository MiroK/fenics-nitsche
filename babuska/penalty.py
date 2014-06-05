'''
Enforcing Dirichlet boundary conditions via penalty term following

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


def penalty_solver(N, penalty):
    'Solve the Poisson problem.'

    mesh = UnitSquareMesh(N, N)

    V = FunctionSpace(mesh, 'CG', 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    h = mesh.ufl_cell().max_facet_edge_length
    sigma = Constant(penalty)
    a = inner(grad(u), grad(v))*dx + (1/h**sigma)*inner(u, v)*ds
    L = inner(f, v)*dx + (1/h**sigma)*inner(u_exact, v)*ds

    A, b = assemble_system(a, L)

    uh = Function(V)
    solve(A, uh.vector(), b)

    plot(uh, interactive=True)

    error_L2 = errornorm(u_exact, uh, 'L2')
    error_H1 = errornorm(u_exact, uh, 'H1')

    return Result(h=mesh.hmin(), H1=error_H1, L2=error_L2)

# -----------------------------------------------------------------------------

norm_type = 'H1'
penalty_value = 2  #[1..10] all okay, f has nice regularity
R = penalty_solver(N=4, penalty=penalty_value)
h_ = R.h
e_ = getattr(R, norm_type)
for N in [8, 16, 32, 64, 128]:
    R = penalty_solver(N=N, penalty=penalty_value)
    h = R.h
    e = getattr(R, norm_type)
    rate = ln(e/e_)/ln(h/h_)
    print '{h:.3E} {e:.3E} {rate:.2f}'.format(h=h, e=e, rate=rate)
