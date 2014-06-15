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

from petsc4py import PETSc
from dolfin import *

u_exact = Expression('x[0]*x[1] + sin(pi*x[0])*cos(2*pi*x[1])')
f = Expression('5*pi*pi*sin(pi*x[0])*cos(2*pi*x[1])')


mesh = UnitSquareMesh(20, 20)

V = FunctionSpace(mesh, 'CG', 1)
u = TrialFunction(V)
v = TestFunction(V)

# Project g onto function space living on the boundary
m = inner(u, v)*ds
L = inner(u_exact, v)*ds

M = PETScMatrix()
b = PETScVector()
assemble(m, tensor=M)
assemble(L, tensor=b)
print b.array()

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

plot(uh)
plot(u_exact, mesh=mesh)
interactive()

