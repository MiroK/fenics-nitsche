'''
    Steady Stokes flow inside rectangle domain with cylinder on centerline with
    two outlets is considered. Velocity is prescribed on inlet, the wall of
    channel. Cylinder boundaries are free-slip. The top oulet has
    stress free condition imposed weakly and the bottom outlet has pressure
    enforced weakly.

    For simplicity, boundary conditions on velocity are always set strongly.
    The free slip conditions

        u.n = 0,
        t - (t.n)n = 0  on cylinder

    are set via penalty method of Babuska,

        [Babuska; The finite element method with lagrangian multipliers]

    or symmetric Nitsche method

        [Freund, Stenberg;
        On Weakly Imposed Boundary Conditions for Second Order Problems.]

    More specifically the stress free part is imposed weakly, wherease the
    impermeability uses penalty/Nitsche.
'''

from dolfin import *

f = Constant((0., 0.))
u_inlet = Expression(('sin(pi*x[1])', '0'))
u_wall = Constant((0., 0.))
u_cylinder = Constant((0., 0.))
p_io = Constant(1000.)
mu = Constant(100)

def penalty():
    'Penalty formulation.'
    mesh = Mesh('../meshes/bifurc_hole.xml')
    boundaries =\
        MeshFunction('size_t', mesh, '../meshes/bifurc_hole_facet_region.xml')

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    M = MixedFunctionSpace([V, Q])

    u, p = TrialFunctions(M)
    v, q = TestFunctions(M)

    # These are all marked boundaries
    inlet_bdry = 15
    wall_bdry = 16
    cylinder_bdry = 17
    outlet0_bdry = 18
    outlet1_bdry = 19

    ds = Measure('ds')[boundaries]
    n = FacetNormal(mesh)

    D = lambda u: sym(grad(u))
    S = lambda u, p: -p*Identity(2) + 2*mu*D(u)
    t = lambda u, p: dot(S(u, p), n)

    sigma = Constant(2)  # penalty parameter
    h = mesh.ufl_cell().max_facet_edge_length

    # Previously we set (-p*I + grad(u)).n = 0 on oulet_0 bdry
    # This leaves the transpose part in stress formulation

    a = inner(2*mu*D(u), D(v))*dx - inner(p, div(v))*dx\
        - inner(q, div(u))*dx - inner(2*mu*dot(D(u), n), v)*ds(outlet1_bdry)\
        - inner(dot(mu*grad(u).T, n), v)*ds(outlet0_bdry)\
        - inner(dot(t(u, p), n), dot(v, n))*ds(cylinder_bdry)\
        + 1./h**sigma*inner(dot(u, n), dot(v, n))*ds(cylinder_bdry)

    L = inner(f, v)*dx - inner(p_io*n, v)*ds(outlet1_bdry)

    # Velocity on rest, strongly
    bc_inlet = DirichletBC(M.sub(0), u_inlet, boundaries, inlet_bdry)
    bc_wall = DirichletBC(M.sub(0), u_wall, boundaries, wall_bdry)
    bcs = [bc_inlet, bc_wall]

    up = Function(M)
    solve(a == L, up, bcs)

    u, p = up.split(True)

    plot(u, title='Classic, velocity')
    plot(p, title='Classic, pressure')
    interactive()


def nitsche():
    'Nitsche formulation.'
    mesh = Mesh('../meshes/bifurc_hole.xml')
    boundaries =\
        MeshFunction('size_t', mesh, '../meshes/bifurc_hole_facet_region.xml')

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    M = MixedFunctionSpace([V, Q])

    u, p = TrialFunctions(M)
    v, q = TestFunctions(M)

    # These are all marked boundaries
    inlet_bdry = 15
    wall_bdry = 16
    cylinder_bdry = 17
    outlet0_bdry = 18
    outlet1_bdry = 19

    ds = Measure('ds')[boundaries]
    n = FacetNormal(mesh)
    h = mesh.ufl_cell().max_facet_edge_length

    beta = Constant(20)  # penalty parameter

    D = lambda u: sym(grad(u))
    S = lambda u, p: -p*Identity(2) + 2*mu*D(u)
    t = lambda u, p: dot(S(u, p), n)

    # Previously we set (-p*I + grad(u)).n = 0 on oulet_0 bdry
    # This leaves the transpose part in stress formulation

    a = inner(2*mu*D(u), D(v))*dx - inner(p, div(v))*dx\
        - inner(q, div(u))*dx - inner(2*mu*dot(D(u), n), v)*ds(outlet1_bdry)\
        - inner(dot(mu*grad(u).T, n), v)*ds(outlet0_bdry)\
        - inner(dot(t(u, p), n), dot(v, n))*ds(cylinder_bdry)\
        - inner(dot(t(v, q), n), dot(u, n))*ds(cylinder_bdry)\
        + beta/h*inner(dot(u, n), dot(v, n))*ds(cylinder_bdry)

    L = inner(f, v)*dx - inner(p_io*n, v)*ds(outlet1_bdry)


    # Velocity on rest, strongly
    bc_inlet = DirichletBC(M.sub(0), u_inlet, boundaries, inlet_bdry)
    bc_wall = DirichletBC(M.sub(0), u_wall, boundaries, wall_bdry)
    bcs = [bc_inlet, bc_wall]

    up = Function(M)
    solve(a == L, up, bcs)
    u, p = up.split(True)

    plot(u, title='Nitsche, velocity')
    plot(p, title='Nitsche, pressure')
    interactive()

# -----------------------------------------------------------------------------

penalty()
nitsche()
