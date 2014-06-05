'''
    Steady Navier-Stokes flow inside rectangle domain with cylinder on cente
    rline with two outlets is considered. Velocity is prescribed on inlet, the
    wall of channel as well as cylincer walls are no slip boundaries. The top
    oulet has stress free condition imposed weakly and the bottom outlet has
    pressure enforced weakly.

    We compare standrard formulation emposing velocity bcs strongly with
    the symmetric Nische formulation that imposes these conditions weakly.
    Moreover Nitsche formulation includes terms that select appropriate
    boundary conditions for strongly convective flows, that is velocity
    prescribed only on inlet, and for inviscid flows where only normal
    component is controlled. These are inspired by

        [Becker;
        Mesh adaptation for Dirichlet flow control via Nitsche's method]
'''



from dolfin import *

f = Constant((0., 0.))
u_inlet = Expression(('sin(pi*x[1])', '0'))
u_wall = Constant((0., 0.))
u_cylinder = Constant((0., 0.))
p_io = Constant(1000.)
mu = Constant(10)


def classic():
    'Velocity bcs enforced on LA level.'
    mesh = Mesh('../meshes/bifurc_hole.xml')
    boundaries =\
        MeshFunction('size_t', mesh, '../meshes/bifurc_hole_facet_region.xml')

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    M = MixedFunctionSpace([V, Q])

    up = TrialFunction(M)
    u, p = split(up)

    vq = TestFunction(M)
    v, q = split(vq)

    up0 = Function(M)
    u0, p0 = split(up0)

    inlet_bdry = 15
    wall_bdry = 16
    cylinder_bdry = 17
    outlet0_bdry = 18
    outlet1_bdry = 19

    ds = Measure('ds')[boundaries]
    n = FacetNormal(mesh)

    F = inner(dot(grad(u), u0), v)*dx + inner(mu*grad(u), grad(v))*dx\
        - inner(p, div(v))*dx - inner(q, div(u))*dx\
        - inner(mu*dot(grad(u), n), v)*ds(outlet1_bdry)\
        - inner(f, v)*dx + inner(p_io*n, v)*ds(outlet1_bdry)

    a, L = system(F)

    bc_inlet = DirichletBC(M.sub(0), u_inlet, boundaries, inlet_bdry)
    bc_wall = DirichletBC(M.sub(0), u_wall, boundaries, wall_bdry)
    bc_cylinder = DirichletBC(M.sub(0), u_cylinder, boundaries, cylinder_bdry)
    bcs = [bc_inlet, bc_wall, bc_cylinder]

    up = Function(M)
    iter = 0
    iter_max = 50
    converged = False
    tol = 1E-8
    e = -1
    while not converged and iter < iter_max:
        iter += 1
        print 'Iteration', iter

        up0.assign(up)

        A = assemble(a)
        b = assemble(L)
        [bc.apply(A, b) for bc in bcs]

        solve(A, up.vector(), b)

        if iter == 1:
            converged = False
            u, _ = up.split(True)
            error0 = sqrt(assemble(inner(u, u)*dx))
        else:
            u, _ = up.split(True)
            error = sqrt(assemble(inner(u, u)*dx))
            e = abs(error - error0)/(error + error0)
            error0 = error
            converged = e < tol

        print '\t Error', e

    u, p = up.split(True)

    plot(u, title='Classic, velocity')
    plot(p, title='Classic, pressure')
    interactive()


def nitsche():
    'Velocity bcs enforced weakly with symmetric Nitsche method.'
    mesh = Mesh('../meshes/bifurc_hole.xml')
    boundaries =\
        MeshFunction('size_t', mesh, '../meshes/bifurc_hole_facet_region.xml')

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    M = MixedFunctionSpace([V, Q])

    u, p = TrialFunctions(M)
    v, q = TestFunctions(M)

    up0 = Function(M)
    u0, p0 = split(up0)

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
    a = inner(dot(grad(u), u0), v)*dx + inner(mu*grad(u), grad(v))*dx\
        - inner(p, div(v))*dx - inner(q, div(u))*dx\
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

    # Finaly we add terms that enforce only normal component of velocity
    # in case the flow is inviscid and terms that set bcs only on the inflow -
    # this is usefull for convection dominnt flows
    beta2 = Constant(10.)
    # Return zero value for outflow
    inflow = lambda u, n: conditional(lt(dot(u, n), 0),
                                      dot(u, n),
                                      Constant(0.))

    a += beta2/h*inner(dot(u, n), dot(v, n))*ds_u\
        - inflow(u0, n)*inner(u, v)*ds_u

    L += beta2/h*inner(dot(u_inlet, n), dot(v, n))*ds(inlet_bdry)\
        - inflow(u0, n)*inner(u_inlet, v)*ds(inlet_bdry)\
        + beta2/h*inner(dot(u_cylinder, n), dot(v, n))*ds(cylinder_bdry)\
        - inflow(u0, n)*inner(u_cylinder, v)*ds(cylinder_bdry)\
        + beta2/h*inner(dot(u_wall, n), dot(v, n))*ds(wall_bdry)\
        - inflow(u0, n)*inner(u_wall, v)*ds(wall_bdry)

    up = Function(M)
    iter = 0
    iter_max = 50
    converged = False
    tol = 1E-8
    e = -1
    while not converged and iter < iter_max:
        iter += 1
        print 'Iteration', iter

        up0.assign(up)

        A = assemble(a)
        b = assemble(L)

        solve(A, up.vector(), b)

        if iter == 1:
            converged = False
            u, _ = up.split(True)
            error0 = sqrt(assemble(inner(u, u)*dx))
        else:
            u, _ = up.split(True)
            error = sqrt(assemble(inner(u, u)*dx))
            e = abs(error - error0)/(error + error0)
            error0 = error
            converged = e < tol

        print '\t Error', e

    u, p = up.split(True)

    plot(u, title='Nitsche, velocity')
    plot(p, title='Nitsche, pressure')
    interactive()

# -----------------------------------------------------------------------------

classic()
nitsche()
