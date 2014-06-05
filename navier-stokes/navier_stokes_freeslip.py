
'''
    This is an extension of navier_stokes_noslip to cylinder with free slip
    boundary conditions.
'''



from dolfin import *

f = Constant((0., 0.))
u_inlet = Expression(('sin(pi*x[1])', '0'))
u_wall = Constant((0., 0.))
u_cylinder = Constant((0., 0.))
p_io = Constant(1000.)
mu = Constant(10)


from dolfin import *

def penalty():
    'Enforcing u.n = 0 by penalty method.'
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

    mu = Constant(10)
    ds = Measure('ds')[boundaries]
    n = FacetNormal(mesh)
    h = mesh.ufl_cell().max_facet_edge_length

    D = lambda u: sym(grad(u))
    S = lambda u, p: -p*Identity(2) + 2*mu*D(u)
    t = lambda u, p: dot(S(u, p), n)
    s = Constant(2)

    F = inner(dot(grad(u), u0), v)*dx + inner(2*mu*D(u), D(v))*dx\
        - inner(p, div(v))*dx - inner(q, div(u))*dx\
        - inner(dot(mu*grad(u).T, n), v)*ds(outlet0_bdry)\
        - inner(2*mu*dot(D(u), n), v)*ds(outlet1_bdry)\
        - inner(dot(t(u, p), n), dot(v, n))*ds(cylinder_bdry)\
        + (1/h**s)*inner(dot(u, n), dot(v, n))*ds(cylinder_bdry)\
        - inner(f, v)*dx + inner(p_io*n, v)*ds(outlet1_bdry)

    a, L = system(F)

    bc_inlet = DirichletBC(M.sub(0), u_inlet, boundaries, inlet_bdry)
    bc_wall = DirichletBC(M.sub(0), u_wall, boundaries, wall_bdry)
    bcs = [bc_inlet, bc_wall]

    up = Function(M)
    iter = 0
    iter_max = 50
    converged = False
    tol = 1E-8
    e = 0
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

    plot(u, title='Penalty, velocity')
    plot(p, title='Penalty, pressure')
    interactive()


def nitsche():
    '''Velocity bcs on wall and inlet as well as u.n=0 on cylinder are enforced
    strongly.'''
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

    D = lambda u: sym(grad(u))
    S = lambda u, p: -p*Identity(2) + 2*mu*D(u)
    t = lambda u, p: dot(S(u, p), n)

    # Volume parts
    a = inner(dot(grad(u), u0), v)*dx\
        - inner(p, div(v))*dx - inner(q, div(u))*dx\
        + inner(2*mu*D(u), D(v))*dx

    L = inner(f, v)*dx

    # Add pressure boundary term
    a += - inner(dot(2*mu*D(u), n), v)*ds(outlet1_bdry)
    L += - inner(p_io*n, v)*ds(outlet1_bdry)

    # Add zero traction on oulet0
    a += - inner(dot(mu*grad(u).T, n), v)*ds(outlet0_bdry)

    beta1 = Constant(40)
    # Add free slip, u.n = 0 so nothing for L
    a += - inner(dot(v, n), dot(t(u, p), n))*ds(cylinder_bdry)\
        - inner(dot(u, n), dot(t(v, q), n))*ds(cylinder_bdry)\
        + beta1*mu/h*inner(dot(u, n), dot(v, n))*ds(cylinder_bdry)

    # Add velocity
    ds_u = ds(inlet_bdry) + ds(wall_bdry)
    a += -inner(dot(S(u, p), n), v)*ds_u\
        - inner(dot(S(v, q), n), u)*ds_u\
        + beta1*mu/h*inner(u, v)*ds_u

    L += - inner(dot(S(v, q), n), u_inlet)*ds(inlet_bdry)\
        + beta1*mu/h*inner(u_inlet, v)*ds(inlet_bdry)\
        - inner(dot(S(v, q), n), u_wall)*ds(wall_bdry)\
        + beta1*mu/h*inner(u_wall, v)*ds(wall_bdry)

    # Add the advection terms
    beta2 = Constant(10.)
    # Return zero value for outflow
    inflow = lambda u, n: conditional(lt(dot(u, n), 0),
                                      dot(u, n),
                                      Constant(0.))

    a += beta2/h*inner(dot(u, n), dot(v, n))*ds_u\
        - inflow(u0, n)*inner(u, v)*ds_u

    L += beta2/h*inner(dot(u_inlet, n), dot(v, n))*ds(inlet_bdry)\
        - inflow(u0, n)*inner(u_inlet, v)*ds(inlet_bdry)\
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

#penalty()
nitsche()
