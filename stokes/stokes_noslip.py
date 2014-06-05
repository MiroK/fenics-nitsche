'''
    Steady Stokes flow inside rectangle domain with cylinder on centerline with
    two outlets is considered. Velocity is prescribed on inlet, the wall of
    channel as well as cylincer walls are no slip boundaries. The top oulet has
    stress free condition imposed weakly and the bottom outlet has pressure
    enforced weakly.

    Standard FEM formulation with strongly enforced velocity boundary
    conditions is compared with symmetric and skew symmetric Nitsche
    formulations. The latter two follow respectively
    [dolfin-adjoin stokes demo] and

    [J. Benk et al.;
    The Nitsche method of the N-S eqs for immersed and moving bdaries].
'''

from dolfin import *

f = Constant((0., 0.))
u_inlet = Expression(('sin(pi*x[1])', '0'))
u_wall = Constant((0., 0.))
u_cylinder = Constant((0., 0.))
p_io = Constant(1000.)
mu = Constant(100)

def classic():
    'Classical formulation.'
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

    # Stress = 0 on oulet0, weakly
    # pressure on outlet1, weakly
    a = inner(mu*grad(u), grad(v))*dx - inner(p, div(v))*dx\
        - inner(q, div(u))*dx - inner(mu*dot(grad(u), n), v)*ds(outlet1_bdry)

    L = inner(f, v)*dx - inner(p_io*n, v)*ds(outlet1_bdry)

    # Velocity on rest, strongly
    bc_inlet = DirichletBC(M.sub(0), u_inlet, boundaries, inlet_bdry)
    bc_wall = DirichletBC(M.sub(0), u_wall, boundaries, wall_bdry)
    bc_cylinder = DirichletBC(M.sub(0), u_cylinder, boundaries, cylinder_bdry)
    bcs = [bc_inlet, bc_wall, bc_cylinder]

    up = Function(M)
    solve(a == L, up, bcs)

    u, p = up.split(True)

    plot(u, title='Classic, velocity')
    plot(p, title='Classic, pressure')
    interactive()


def nitsche(symmetric=True):
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

    # Define forms
    # Note that (skew)-symmetry is both in `standard`=dx terms
    # and also Nitsche ones.
    if symmetric:
        a = inner(mu*grad(u), grad(v))*dx - inner(p, div(v))*dx\
            - inner(q, div(u))*dx\
            - inner(mu*dot(grad(u), n), v)*ds(outlet1_bdry)

        L = inner(f, v)*dx - inner(p_io*n, v)*ds(outlet1_bdry)
        # At this point, we included pressure and stress bcs weakly

        # Add velocity bcs weakly via symmetric Nitsche
        beta = Constant(20.)

        ds_u = ds(inlet_bdry) + ds(wall_bdry) + ds(cylinder_bdry)
        a += inner(p*n, v)*ds_u\
            - inner(dot(mu*grad(u), n), v)*ds_u\
            + inner(q*n, u)*ds_u\
            - inner(dot(mu*grad(v), n), u)*ds_u\
            + mu*beta/h*inner(u, v)*ds_u

        L += inner(q*n, u_inlet)*ds(inlet_bdry)\
            - inner(dot(mu*grad(v), n), u_inlet)*ds(inlet_bdry)\
            + mu*beta/h*inner(u_inlet, v)*ds(inlet_bdry)\
            + inner(q*n, u_wall)*ds(wall_bdry)\
            + inner(q*n, u_wall)*ds(wall_bdry)\
            - inner(dot(mu*grad(v), n), u_wall)*ds(wall_bdry)\
            + mu*beta/h*inner(u_cylinder, v)*ds(cylinder_bdry)\
            - inner(dot(mu*grad(v), n), u_cylinder)*ds(cylinder_bdry)\
            + mu*beta/h*inner(u_cylinder, v)*ds(cylinder_bdry)
    else:
        # The basis for has B and -B^T
        # The bc enforcing form as pressure parts with oposing signs. In addi-
        # tion there is an additional stabilizing term (more relevant for
        # zero viscosity limit)
        a = mu*inner(grad(u), grad(v))*dx - inner(p, div(v))*dx\
            + inner(q, div(u))*dx\
            - inner(mu*dot(grad(u), n), v)*ds(outlet1_bdry)

        L = inner(f, v)*dx - inner(p_io*n, v)*ds(outlet1_bdry)
        # At this point, we included pressure and stress bcs weakly

        gamma_1 = Constant(20.)
        gamma_2 = Constant(20.)
        ds_u = ds(inlet_bdry) + ds(wall_bdry) + ds(cylinder_bdry)

        a += inner(p*n, v)*ds_u\
            - inner(dot(mu*grad(u), n), v)*ds_u\
            - inner(q*n, u)*ds_u\
            - inner(dot(mu*grad(v), n), u)*ds_u\
            + mu*gamma_1/h*inner(u, v)*ds_u\
            + gamma_2/h*inner(dot(u, n), dot(v, n))*ds_u

        L += -inner(q*n, u_inlet)*ds(inlet_bdry)\
            - inner(dot(mu*grad(v), n), u_inlet)*ds(inlet_bdry)\
            + mu*gamma_1/h*inner(u_inlet, v)*ds(inlet_bdry)\
            + gamma_2/h*inner(dot(u_inlet, n), dot(v, n))*ds(inlet_bdry)\
            - inner(q*n, u_wall)*ds(wall_bdry)\
            - inner(dot(mu*grad(v), n), u_wall)*ds(wall_bdry)\
            + mu*gamma_1/h*inner(u_wall, v)*ds(wall_bdry)\
            + gamma_2/h*inner(dot(u_wall, n), dot(v, n))*ds(wall_bdry)\
            - inner(q*n, u_cylinder)*ds(cylinder_bdry)\
            - inner(dot(mu*grad(v), n), u_cylinder)*ds(cylinder_bdry)\
            + mu*gamma_1/h*inner(u_cylinder, v)*ds(cylinder_bdry)\
            + gamma_2/h*inner(dot(u_cylinder, n), dot(v, n))*ds(cylinder_bdry)

    up = Function(M)
    solve(a == L, up)

    u, p = up.split(True)

    symmetry = 'symmetric' if symmetric else 'skew-symmetric'
    plot(u, title='Nitsche %s, velocity' % symmetry)
    plot(p, title='Nitsche %s, pressure' % symmetry)
    interactive()

# -----------------------------------------------------------------------------

classic()
nitsche(symmetric=True)
nitsche(symmetric=False)
