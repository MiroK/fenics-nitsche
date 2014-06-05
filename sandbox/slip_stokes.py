'''
    Steady pressure driven flow through square with cylinder on centerline.
    The cylinder is super greasy and allows free-slip conditions

    u.n = 0
    t - (t.n).n = 0, on the cylinder. t is stress force, dot(S, n).

    The condition is set using symmetric Nitsche formulation following
    [Freund, Stenberg;
     On Weakly Imposed Boundary Conditions for Second Order Problems.]
'''

from dolfin import *

f = Constant((0., 0.))
g_wall = Constant((0., 0.))
un_cylinder = Constant(0.)
h_io = Expression('4 - x[0]')


def nitsche_stokes(mu_viscosity):
    'Nitsche formulation.'

    mesh = Mesh('../meshes/square_hole.xml')
    boundaries =\
        MeshFunction('size_t', mesh, '../meshes/square_hole_facet_region.xml')

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    M = MixedFunctionSpace([V, Q])

    u, p = TrialFunctions(M)
    v, q = TestFunctions(M)

    pressure_bdry = 13
    wall_bdry = 12
    cylinder_bdry = 14

    mu = Constant(mu_viscosity)
    ds = Measure('ds')[boundaries]
    n = FacetNormal(mesh)
    h_E = mesh.ufl_cell().max_facet_edge_length
    beta = Constant(1000)

    D = lambda u: sym(grad(u))
    S = lambda u, p: -p*Identity(2) + 2*mu*D(u)
    t = lambda u, p: dot(S(u, p), n)

    a = 2*mu*inner(D(u), D(v))*dx -\
        inner(p, div(v))*dx - inner(q, div(u))*dx -\
        2*mu*inner(dot(D(u), n), v)*ds(pressure_bdry)\

    L = inner(f, v)*dx - inner(h_io*n, v)*ds(pressure_bdry)
    # So far pressure bcs are included weakly

    # Weakly include conditions t = (t.n)n
    a += -inner(dot(t(u, p), n), dot(v, n))*ds(cylinder_bdry)

    # Weakly include u.n = 0
    # The terms are symmetry, stability...
    a += -inner(dot(u, n), dot(t(v, q),n))*ds(cylinder_bdry)\
        + beta*h_E**-1*inner(dot(u, n), dot(v, n))*ds(cylinder_bdry)
    # ... consistency
    L += -inner(un_cylinder, dot(t(v, q), n))*ds(cylinder_bdry)\
        + beta*h_E**-1*inner(un_cylinder, dot(v, n))*ds(cylinder_bdry)

    # Wall bcs are set strongly
    bc = DirichletBC(M.sub(0), g_wall, boundaries, wall_bdry)

    A, b = assemble_system(a, L, bc)
    up = Function(M)
    solve(A, up.vector(), b)

    u, p = up.split()
    plot(u)
    plot(p)
    interactive()

nitsche_stokes(mu_viscosity=1)
