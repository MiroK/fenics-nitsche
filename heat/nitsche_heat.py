from dolfin import *

# Problem parameters
diffusion = 1
T0 = 1  # Left
T1 = 1  # Bottom
T2 = 1  # Right
T3 = 1  # Top
time_step = 0.001
T_final = 0.5
eps = 0.25     # Half-width of the heated zone


def left(x, on_boundary):
    return on_boundary and near(x[0], 0)


def right(x, on_boundary):
    return on_boundary and near(x[0], 1)


def bottom(x, on_boundary):
    return on_boundary and near(x[1], 0)


def top(x, on_boundary):
    return on_boundary and near(x[1], 1)


class BoundarySource(Expression):
    def __init__(self, magnitude, i):
        self.mag = magnitude
        self.i = i

    def eval(self, values, x):
        if abs(x[self.i] - 0.5) < eps + DOLFIN_EPS:
            values[0] = self.mag
        else:
            values[0] = 0

mesh = UnitSquareMesh(40, 40)
V = FunctionSpace(mesh, 'CG', 1)

u = TrialFunction(V)
v = TestFunction(V)

kappa = Constant(diffusion)
dt = Constant(time_step)

boundaries = FacetFunction('size_t', mesh, 0)
AutoSubDomain(left).mark(boundaries, 1)
AutoSubDomain(bottom).mark(boundaries, 2)
AutoSubDomain(right).mark(boundaries, 3)
AutoSubDomain(top).mark(boundaries, 4)
ds = Measure('ds')[boundaries]
gamma = Constant(4)
h_E = mesh.ufl_cell().max_facet_edge_length
n = FacetNormal(mesh)
g1 = BoundarySource(T0/eps, 1)
g2 = BoundarySource(T1/eps, 0)
g3 = BoundarySource(T2/eps, 1)
g4 = BoundarySource(T3/eps, 0)

u0 = Function(V)
u_cn = 0.5*(u + u0)
F = dt**-1*inner(u - u0, v)*dx + inner(kappa*grad(u_cn), grad(v))*dx\
    - inner(dot(kappa*grad(u_cn), n), v)*ds('everywhere')\
    - inner(dot(kappa*grad(v), n), u_cn)*ds('everywhere')\
    + gamma/h_E*inner(u_cn, v)*ds('everywhere')\
    + inner(dot(kappa*grad(v), n), g1)*ds(1)\
    - gamma/h_E*inner(g1, v)*ds(1)\
    + inner(dot(kappa*grad(v), n), g2)*ds(2)\
    - gamma/h_E*inner(g2, v)*ds(2)\
    + inner(dot(kappa*grad(v), n), g3)*ds(3)\
    - gamma/h_E*inner(g3, v)*ds(3)\
    + inner(dot(kappa*grad(v), n), g4)*ds(4)\
    - gamma/h_E*inner(g4, v)*ds(4)\

a, L = system(F)

A = assemble(a)

t = 0
T = T_final
uh = Function(V)
while t < 0.5:
    t += dt(0)
    print t

    b = assemble(L)

    solve(A, uh.vector(), b)
    u0.assign(uh)
    plot(u0)

print u0(0.5, 0.5)
interactive()
