'''
    Steady pressure driven flow through rectangle with cylinder on centerline.

    Standard FEM formulation setting velocity boundary conditions strongly
    is compared with Nitsche method that roughly follows

            [J. Benk et al.;
            The Nitsche method of the N-S eqs for immersed and moving bdaries].
'''

from dolfin import *

f = Constant((0., 0.))
g_wall = Constant((0., 0.))
g_cylinder = Expression(('A*(x[1] - 0.5)', 'A*(2 - x[0])'), A=0)
h_io = Expression('4 - x[0]')


def standard_stokes(reynolds_number):
    'Standard formulation enforcing velocity boundary conditions stongly.'

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

    Re = Constant(reynolds_number)
    ds = Measure('ds')[boundaries]
    n = FacetNormal(mesh)

    a = Re**-1*inner(grad(u), grad(v))*dx -\
        inner(p, div(v))*dx - inner(q, div(u))*dx -\
        Re**-1*inner(dot(grad(u), n), v)*ds(pressure_bdry)

    L = inner(f, v)*dx - inner(h_io*n, v)*ds(pressure_bdry)

    bc_wall = DirichletBC(M.sub(0), g_wall, boundaries, wall_bdry)
    bc_cylinder = DirichletBC(M.sub(0), g_cylinder, boundaries, cylinder_bdry)
    bcs_u = [bc_wall, bc_cylinder]

    A, b = assemble_system(a, L, bcs_u)
    up = Function(M)
    solve(A, up.vector(), b)

    u, p = up.split()
    plot(u)
    plot(p)
    interactive()


def nitsche_stokes(reynolds_number, dirichlet_wall=False):
    '''Nitsche formulation enforcing. Velocity boundary conditions
    on cylinder are enforced weakly. The others are part of TrialSpace.'''

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

    Re = Constant(reynolds_number)
    ds = Measure('ds')[boundaries]
    n = FacetNormal(mesh)
    h_E = mesh.ufl_cell().max_facet_edge_length
    gamma_1 = Constant(10)
    gamma_2 = Constant(10)

    # Note +inner(q, div(u))*dx
    a = Re**-1*inner(grad(u), grad(v))*dx -\
        inner(p, div(v))*dx + inner(q, div(u))*dx -\
        Re**-1*inner(dot(grad(u), n), v)*ds(pressure_bdry)\
        - Re**-1*inner(dot(grad(u), n), v)*ds(cylinder_bdry)\
        + inner(p*n, v)*ds(cylinder_bdry)

    L = inner(f, v)*dx - inner(h_io*n, v)*ds(pressure_bdry)

    # Terms to set velocity bcs on cylinder
    # Note - inner(q*n, u)*ds      [vs +inner(p*n, v)*ds]
    # Note gamma_2 forces u.n even with Re=oo (inviscid limit, Euler)
    a += - Re**-1*inner(dot(grad(v), n), u)*ds(cylinder_bdry)\
        - inner(q*n, u)*ds(cylinder_bdry)\
        + Re**-1*gamma_1/h_E*inner(u, v)*ds(cylinder_bdry)\
        + gamma_2/h_E*inner(dot(u, n), dot(v, n))*ds(cylinder_bdry)

    L += - Re**-1*inner(dot(grad(v), n), g_cylinder)*ds(cylinder_bdry)\
        - inner(q*n, g_cylinder)*ds(cylinder_bdry)\
        + Re**-1*gamma_1/h_E*inner(g_cylinder, v)*ds(cylinder_bdry)\
        + gamma_2/h_E*inner(dot(g_cylinder, n), dot(v, n))*ds(cylinder_bdry)

    # Bcs on wall can be set strongly or weakly
    if dirichlet_wall:
        bc_wall = DirichletBC(M.sub(0), g_wall, boundaries, wall_bdry)
        bcs_u = [bc_wall]

        A, b = assemble_system(a, L, bcs_u)
    else:
        a += - Re**-1*inner(dot(grad(u), n), v)*ds(wall_bdry)\
            + inner(p*n, v)*ds(wall_bdry)\
            - Re**-1*inner(dot(grad(v), n), u)*ds(wall_bdry)\
            - inner(q*n, u)*ds(wall_bdry)\
            + Re**-1*gamma_1/h_E*inner(u, v)*ds(wall_bdry)\
            + gamma_2/h_E*inner(dot(u, n), dot(v, n))*ds(wall_bdry)

        L += - Re**-1*inner(dot(grad(v), n), g_wall)*ds(wall_bdry)\
            - inner(q*n, g_wall)*ds(wall_bdry)\
            + Re**-1*gamma_1/h_E*inner(g_wall, v)*ds(wall_bdry)\
            + gamma_2/h_E*inner(dot(g_wall, n), dot(v, n))*ds(wall_bdry)

        A, b = assemble_system(a, L)

    up = Function(M)
    solve(A, up.vector(), b)

    u, p = up.split()
    plot(u)
    plot(p)
    interactive()

standard_stokes(reynolds_number=10)
nitsche_stokes(reynolds_number=10)
