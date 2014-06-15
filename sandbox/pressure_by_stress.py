'''
    As setting pressure is supposed to be difficult on mixed solvers, this
    script investigates alternative formulation where pressure is set via
    requirements on stress. We follow

    [Barth, Carey;
    On a boundary condition for pressure-driven laminar flow of incompressible
    fluids
    ]

    and solve simple pressure driven flow.
'''

from dolfin import *

# Inlet-outlet pressure
p_io = Expression('1 - x[0]')

# Define domains
def top_bottom(x, on_boundary):
    return near(x[1]*(1 - x[1]), 0)


def left_right(x, on_boundary):
    return near(x[0]*(1 - x[0]), 0)

TopBottom = AutoSubDomain(top_bottom)
LeftRight = AutoSubDomain(left_right)

def solver(N):
    'Solve Stokes pressure driven flow on mesh with N ...'
    # The problem
    mesh = UnitSquareMesh(N, N)
    boundaries = FacetFunction('size_t', mesh, 0)
    TopBottom.mark(boundaries, 1)
    LeftRight.mark(boundaries, 2)

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    M = MixedFunctionSpace([V, Q])

    bc = DirichletBC(M.sub(0), Constant((0., 0.)), boundaries, 1)

    u, p = TrialFunctions(M)
    v, q = TestFunctions(M)

    f = Constant((0., 0.))
    mu = Constant(1.)
    D = lambda u: sym(grad(u))
    S = lambda u, p: -p*Identity(2) + 2*mu*D(u)

    # WeakForm
    h = Circumradius(mesh)
    n = FacetNormal(mesh)
    ds = Measure('ds')[boundaries]

    F = inner(S(u, p), D(v))*dx + inner(p_io*n, v)*ds(2)\
        - inner(div(u), q)*dx - inner(f, v)*dx\
        + h**-2*(inner(u, v) - inner(dot(u, n), dot(v, n)))*ds(2)

    a, L = system(F)

    up = Function(M)
    solve(a == L, up, bc)

    u, p = up.split(True)

    plot(u)
    plot(p)
    interactive()

# -----------------------------------------------------------------------------

for N in [8, 16, 32, 64]:
    solver(N=N)
