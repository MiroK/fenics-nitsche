'''
Fractional step for mixed formulation of Navier-Stokes. The problem
is split into 2 Stokes-type and 1 Burgers-type problems. Let [t_n, t_n+1] be
time interval with length dt and

    theta=1-sqrt(2)/2
    alpha = (1-2*theta)/(1-theta)
    beta = 1 - alpha

The 3 problems are
Stokes:
    (u - u0)/theta/dt - alpha/Re*laplace(u) + grad(p)=\
        f^{n+theta} + beta/Re*laplace(u0) - grad(u0)*u0

    div(u) = 0
    u = g^{n+theta}   // Let the solution of this sytem be U0, P0

Burgers:
    (u - U0)/(1-2*theta)/dt - beta/Re*laplace(u) + grad(u)*u=\
        f^{n+1-theta} + alpha/Re*laplace(U0) - grad(P0)

    u = g^{n+1-theta}  // Let the solution of this system be U1

Stokes:
    (u - U1)/theta/dt - alpha/Re*laplace(u) + grad(p)=\
        f^{n+theta} + beta/Re*laplace(U1) - grad(U1)*U1

    div(u) = 0
    u = g^{n+1}
'''

from dolfin import *
set_log_level(WARNING)

# Flow past a cylinder
# The cylinder and top/bottom walls are no slip. Inlet has velocity prescribed
# and outflow is stress free. The conditions result in absence of surface terms
# in integration by parts
# Test case parameters
U_m = 0.5
H = 0.41
T_final = 8  # Period

f = Constant((0., 0.))
u_inlet = Expression(('4*U_m*x[1]*(H-x[1])*sin(pi*t/T_final)/H/H', '0'),
                     U_m=U_m, H=H, T_final=T_final, t=0)
u_wall = Constant((0., 0.))
Re = Constant(100)


def mixed_solver(mesh_):
    'Coupled solver'
    mesh = Mesh('../../meshes/%s.xml' % mesh_)
    boundaries =\
        MeshFunction('size_t', mesh, '../../meshes/%s_facet_region.xml' % mesh_)

    # These are all marked boundaries
    inlet_bdry = 14
    wall_bdry = 13  # includes cylinder
    outlet_bdry = 15

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    M = MixedFunctionSpace([V, Q])

    u, p = TrialFunctions(M)
    v, q = TestFunctions(M)

    # Solution at previous time level
    up0 = Function(M)
    u0, p0 = split(up0)

    # Solution of Stokes problems
    UP0 = Function(M)
    U0, P0 = split(UP0)

    # Solution of Burger problem
    UP1 = Function(M)       # As mixed
    U1, P1 = split(UP1)
    U1_ = Function(V)       # As V

    dt = Constant(0.75*mesh.hmin()/U_m)
    theta = Constant(1-sqrt(2)/2)
    alpha = (1-2*theta)/(1-theta)
    beta = 1 - alpha

    # Form of first Stokes problem
    F0 = (1/theta/dt)*inner(u - u0, v)*dx + alpha/Re*inner(grad(u), grad(v))*dx\
        - inner(p, div(v))*dx - inner(q, div(u))*dx - inner(f, v)*dx\
        + beta/Re*inner(grad(u0), grad(v))*dx + inner(dot(grad(u0), u0), v)*dx
    a0, L0 = system(F0)

    # Form of Burgers problem
    U0_ = Function(V)
    P0_ = Function(Q)

    u_ = TrialFunction(V)
    w = TestFunction(V)
    u_k = Function(V)

    F1 = (1/(1-2*theta)/dt)*inner(u_ - U0_, w)*dx\
        + inner(dot(grad(u_), u_k), w)*dx\
        + beta/Re*inner(grad(u_), grad(w))*dx\
        - inner(f, w)*dx + alpha/Re*inner(grad(U0_), grad(w))*dx\
        - inner(P0_, div(w))*dx
    a1, L1 = system(F1)

    # Form of second Stokes problem
    F2 = (1/theta/dt)*inner(u - U1, v)*dx + alpha/Re*inner(grad(u), grad(v))*dx\
        - inner(p, div(v))*dx - inner(q, div(u))*dx - inner(f, v)*dx\
        + beta/Re*inner(grad(U1), grad(v))*dx + inner(dot(grad(U1), U1), v)*dx
    a2, L2 = system(F2)

    # Boundary conditions for Stokes
    bc_inlet = DirichletBC(M.sub(0), u_inlet, boundaries, inlet_bdry)
    bc_wall = DirichletBC(M.sub(0), u_wall, boundaries, wall_bdry)
    bcs_u = [bc_inlet, bc_wall]

    # Boundary conditions for Burgers
    bc_inlet = DirichletBC(V, u_inlet, boundaries, inlet_bdry)
    bc_wall = DirichletBC(V, u_wall, boundaries, wall_bdry)
    BCS_u = [bc_inlet, bc_wall]

    comm = mesh.mpi_comm()
    u_file = XDMFFile(comm, 'results/u.xdmf'.format(mesh_))
    p_file = XDMFFile(comm, 'results/p_{0}.xdmf'.format(mesh_))
    u_file.parameters['rewrite_function_mesh'] = False
    p_file.parameters['rewrite_function_mesh'] = False

    u_plot = Function(V)
    p_plot = Function(Q)

    up = Function(M)
    t, step = 0, 0

    M0_dofs = M.sub(0).dofmap().dofs()
    M1_dofs = M.sub(1).dofmap().dofs()

    while t < T_final:
        step += 1
        print step

        # Solve first Stokes
        t += dt(0)*theta(0)
        u_inlet.t = t
        solve(a0 == L0, UP0, bcs_u)
        print '\t', t

        # Solve Burgers
        U0_.vector()[:] = UP0.vector()[M0_dofs]
        P0_.vector()[:] = UP0.vector()[M1_dofs]

        t += dt(0)*(1-2*theta(0))
        u_inlet.t = t

        iter = 0
        iter_max = 40
        converged = False
        tol = 1E-4
        e = -1
        while not converged and iter < iter_max:
            iter += 1
            u_k.assign(U1_)

            solve(a1 == L1, U1_, BCS_u)

            if iter == 1:
                converged = False
                error0 = sqrt(assemble(inner(U1_, U1_)*dx))
            else:
                error = sqrt(assemble(inner(U1_, U1_)*dx))
                e = abs(error - error0)/(error + error0)
                error0 = error
                converged = e < tol

            print '\t\t iter {0}, error {1}'.format(iter, e)
        print '\t', t

        # Solver Stokes
        UP1.vector()[M0_dofs] = U1_.vector()

        t += dt(0)*theta(0)
        u_inlet.t = t
        solve(a2 == L2, up0, bcs_u)
        print '\t', t

        # Store
        if step % 10 == 0:
            u_plot.assign(up0.split(True)[0])
            p_plot.assign(up0.split(True)[1])

            plot(u_plot, title='Velocity @ t = {0}'.format(t))
            plot(p_plot, title='Pressure @ t = {0}'.format(t))

            u_file << u_plot
            p_file << p_plot

    interactive()

# -----------------------------------------------------------------------------

mixed_solver('turek_1')
