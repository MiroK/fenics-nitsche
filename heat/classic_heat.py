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

mesh = UnitSquareMesh(20, 20)
V = FunctionSpace(mesh, 'CG', 1)

u = TrialFunction(V)
v = TestFunction(V)

kappa = Constant(diffusion)
dt = Constant(time_step)

u0 = Function(V)
u_cn = 0.5*(u + u0)
F = dt**-1*inner(u - u0, v)*dx + inner(kappa*grad(u_cn), grad(v))*dx
a, L = system(F)

bc_l = DirichletBC(V, BoundarySource(T0/eps, 1), left)
bc_b = DirichletBC(V, BoundarySource(T1/eps, 0), bottom)
bc_r = DirichletBC(V, BoundarySource(T2/eps, 1), right)
bc_t = DirichletBC(V, BoundarySource(T3/eps, 0), top)
bcs = [bc_l, bc_b, bc_r, bc_t]

A = assemble(a)

t = 0
T = T_final
uh = Function(V)
while t < 0.5:
    t += dt(0)
    print t

    b = assemble(L)
    [bc.apply(A, b) for bc in bcs]

    solve(A, uh.vector(), b)
    u0.assign(uh)
    plot(u0)

print u0(0.5, 0.5)
interactive()
