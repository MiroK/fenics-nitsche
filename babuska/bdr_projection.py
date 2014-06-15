'''
    Solve:
        Find u in S such that <u, s> = <g, s> for all s in S.
        S is a function space defined on the boundary, s are its basis
        functions. I want to test if the projection form has to be changed
        based on the shape of the boundary makes.
'''

from petsc4py import PETSc
from dolfin import *

g = Expression('sin(2*pi*(x[0]*x[0]/2 + x[1]*x[1]))')

def bdr_project(mesh):
    # Global function space
    V = FunctionSpace(mesh, 'CG', 1)

    # Get dofs living on bdry = dofs of S
    bc = DirichletBC(V, Constant(0), DomainBoundary())
    bdr_dofs = bc.get_boundary_values().keys()
    bdr_dofs.sort()
    bdr_dofs_petsc = PETSc.IS()
    bdr_dofs_petsc.createGeneral(bdr_dofs)

    # Projection form
    u = TrialFunction(V)
    v = TestFunction(V)
    mV = inner(u, v)*ds
    LV = inner(g, v)*ds

    # Global projection system
    MV = PETScMatrix()
    bV = PETScVector()
    assemble(mV, tensor=MV)
    assemble(LV, tensor=bV)

    # Bdr projection system
    M_petsc = PETSc.Mat()
    b_petsc = PETSc.Vec()
    MV.mat().getSubMatrix(bdr_dofs_petsc, bdr_dofs_petsc, M_petsc)
    bV.vec().getSubVector(bdr_dofs_petsc, b_petsc)
    M = PETScMatrix(M_petsc)
    b = PETScVector(b_petsc)

    # Vector to hold projected values
    x_petsc = PETSc.Vec()
    bV.vec().getSubVector(bdr_dofs_petsc, x_petsc)
    x = PETScVector(x_petsc)
    solve(M, x, b)

    # Fill projected g
    gS = Function(V)
    gS.vector()[bdr_dofs] = x

    # Build function for comparison by global projection g and setting
    # inner dofs to 0
    gV = project(g, V)
    y = gV.vector().get_local()
    inner_dofs = filter(lambda d: d not in bdr_dofs, range(V.dim()))
    y[inner_dofs] = 0
    gV.vector().set_local(y)
    gV.vector().apply('insert')

    plot(gV)
    plot(gS)
    interactive()

    # Compute the L2(bdry) error
    error = sqrt(assemble(inner(gV - gS, gV - gS)*ds))
    return error

# -----------------------------------------------------------------------------

mesh = Mesh('circle.xml')
n_refinements = 5
for n in range(n_refinements):
    mesh = refine(mesh)
    print bdr_project(mesh)
