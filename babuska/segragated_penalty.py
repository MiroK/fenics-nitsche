'''
We want to solve Poisson problem:

    -laplace(u) = f in \Omega
              u = g on \partial\Omega

with the method descibed in
    [Gunzburger and Hou,
     Treating inhomogeneous essential boundary conditions in
     finite element methods and the calculation of boundary stresses]

The method build on FEM with Lagrange multilpier by Babuska but splits the
saddle point problem into 3 systems. The first one projects g on function
space defined on the boundary. The second one takes projected g and uses
as a stong bc to solve the Poisson problem. Finally the lagrange multiplies
is obtained. The advantage is that one is always solving smaller problems
than the saddle point problem.

We are intersted in convergence of u and the lagrange multiplier t.
'''

from math import log as ln
from petsc4py import PETSc
from dolfin import *

set_log_level(WARNING)

class H(Expression):
    def eval(self, values, x):
        # Top
        if near(x[1], 1) and between(x[0], (0, 1)):
            values[0] = x[0] - 2*pi*sin(pi*x[0])*sin(2*pi*x[1])
        # Bottom
        elif near(x[1], 0) and between(x[0], (0, 1)):
            values[0] = -x[0] + 2*pi*sin(pi*x[0])*sin(2*pi*x[1])
        # Left
        elif near(x[0], 0) and between(x[1], (0, 1)):
            values[0] = -x[1] - pi*cos(pi*x[0])*cos(2*pi*x[1])
        # Right
        elif near(x[0], 1) and between(x[1], (0, 1)):
            values[0] = x[1] + pi*cos(pi*x[0])*cos(2*pi*x[1])
        else:
            values[0] = 0.


u_exact = Expression('x[0]*x[1] + sin(pi*x[0])*cos(2*pi*x[1])')
f = Expression('5*pi*pi*sin(pi*x[0])*cos(2*pi*x[1])')
h_exact = H()

def solver(N, p):
    'Solve the problem and return error norm of u and Lambda.'
    mesh = UnitSquareMesh(N, N)

    V = FunctionSpace(mesh, 'CG', p)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Project g onto function space living on the boundary
    m = inner(u, v)*ds
    L = inner(u_exact, v)*ds

    M = PETScMatrix()
    b = PETScVector()
    assemble(m, tensor=M)
    assemble(L, tensor=b)

    # Get the boundary dofs, ie dofs of bdr FunctionSpace
    g = Function(V)
    bc = DirichletBC(V, g, DomainBoundary())
    bc_dofs = bc.get_boundary_values().keys()
    bc_dofs.sort()

    # Extract the projection submatrix, subvector as PETSc objetsc
    bc_dofs_petsc = PETSc.IS()
    bc_dofs_petsc.createGeneral(bc_dofs)

    M_petsc = PETSc.Mat()
    b_petsc = PETSc.Vec()

    M.mat().getSubMatrix(bc_dofs_petsc, bc_dofs_petsc, M_petsc)
    b.vec().getSubVector(bc_dofs_petsc, b_petsc)

    # Convert to dolfin objects
    M_proj = PETScMatrix(M_petsc)
    b_proj = PETScVector(b_petsc)

    # Create vector living on the bdry to solve the projection problem
    g_petsc = PETSc.Vec()
    b.vec().getSubVector(bc_dofs_petsc, g_petsc)  # to get same size
    g_proj = PETScVector(g_petsc)

    # Solve the projection problem
    solve(M_proj, g_proj, b_proj)

    # Assign projected g to global g
    g.vector()[bc_dofs] = g_proj

    # Solve the Poisson problem with g
    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    uh = Function(V)
    solve(a == L, uh, bc)

    # Asseble rhs for Lagrange multiplier
    assemble(action(a, uh) - L, tensor=b)
    b.vec().getSubVector(bc_dofs_petsc, b_petsc)
    b_Lambda = PETScVector(b_petsc)

    # Solve the problem for Lagrange multiplier
    solve(M_proj, g_proj, b_Lambda)
    Lambda = Function(V)
    Lambda.vector()[bc_dofs] = g_proj

    # plot(uh)
    # plot(u_exact, mesh=mesh)
    # plot(Lambda)
    # plot(h_exact, mesh=mesh)
    # interactive()

    DG = FunctionSpace(mesh, 'DG', p+3)
    u = interpolate(u_exact, DG)
    uh = interpolate(uh, DG)
    eu = u - uh
    eu_norm = sqrt(assemble(inner(eu, eu)*dx))

    h = interpolate(h_exact, DG)
    Lambda = interpolate(Lambda, DG)
    eLambda = h - Lambda
    eLambda_norm = sqrt(assemble(inner(eLambda, eLambda)*ds))

    return mesh.hmin(), eu_norm, eLambda_norm

# -----------------------------------------------------------------------------

p = 2
h_, eu_, eLambda_ = solver(N=8, p=p)
for N in [16, 32, 64, 128]:
    h, eu, eLambda = solver(N=N, p=p)
    rateu = ln(eu/eu_)/ln(h/h_)
    rateLambda = ln(eLambda/eLambda_)/ln(h/h_)

    print h, rateu, rateLambda

# I am not getting incrase in Lambda convergence with p but okay
# not my method
