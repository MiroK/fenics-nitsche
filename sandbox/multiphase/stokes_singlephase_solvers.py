'''
    Smolianski's multiphase solver is built on top of fractional scheme for
    solving the Navier-Stokes problem. The scheme is not the usual IPCS but
    is due to Yanenko.

    Here I'd like to make myself familiar with it by first using it to solve
    transient Stokes problem. The scheme is compared with IPCS and mixed
    solvers.
'''

from dolfin import *

# Test case parameters
U_m = 0.5
H = 0.41
T_final = 8  # Period

f = Constant((0., 0.))
u_inlet = Expression(('4*U_m*x[1]*(H-x[1])*sin(pi*t/T_final)/H/H', '0'),
                     U_m=U_m, H=H, T_final=T_final, t=0)
u_wall = Constant((0., 0.))
mu = Constant(1E-3)
rho = Constant(1)

# Deviatoric stress
D = lambda u: sym(grad(u))

# Stress tensor
S = lambda u, p, mu: -p*Identity(2) + 2*mu*D(u)


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

    dt = Constant(0.75*mesh.hmin()/U_m)

    # Weak form
    up0 = Function(M)
    u0, p0 = split(up0)
    U = 0.5*(u + u0)

    F = (rho/dt)*inner(u - u0, v)*dx - inner(p, div(v))*dx\
        - inner(q, div(u))*dx + inner(2*mu*D(U), D(v))*dx\
        - inner(rho*f, v)*dx
    a, L = system(F)

    # Boundary conditions
    bc_inlet = DirichletBC(M.sub(0), u_inlet, boundaries, inlet_bdry)
    bc_wall = DirichletBC(M.sub(0), u_wall, boundaries, wall_bdry)
    bcs_u = [bc_inlet, bc_wall]

    comm = mesh.mpi_comm()
    u_file = XDMFFile(comm, 'u_stokes_mixed_{0}.xdmf'.format(mesh_))
    p_file = XDMFFile(comm, 'p_stokes_mixed_{0}.xdmf'.format(mesh_))
    u_file.parameters['rewrite_function_mesh'] = False
    p_file.parameters['rewrite_function_mesh'] = False

    u_plot = Function(V)
    p_plot = Function(Q)

    up = Function(M)
    t, step = 0, 0
    while t < T_final:
        t += dt(0)
        step += 1
        print 'step {0}, time {1}'.format(step, t)

        # Update inlet
        u_inlet.t = t

        # Solve predictor step
        solve(a == L, up, bcs_u)
        up0.assign(up)

        # Store
        if step % 10 == 0:
            u_plot.assign(up0.split(True)[0])
            p_plot.assign(up0.split(True)[1])

            plot(u_plot, title='Velocity @ t = {0}'.format(t))
            plot(p_plot, title='Pressure @ t = {0}'.format(t))

            u_file << u_plot
            p_file << p_plot

    interactive()
def ipcs_solver(mesh_):
    'Segregated solver using IPCS'
    mesh = Mesh('../../meshes/%s.xml' % mesh_)
    boundaries =\
        MeshFunction('size_t', mesh, '../../meshes/%s_facet_region.xml' % mesh_)

    # These are all marked boundaries
    inlet_bdry = 14
    wall_bdry = 13  # includes cylinder
    outlet_bdry = 15

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    p = TrialFunction(Q)
    q = TestFunction(Q)

    dt = Constant(0.75*mesh.hmin()/U_m)

    # Form for predicted velocity, solve to U
    u0 = Function(V)
    p0 = Function(Q)
    U_cn = 0.5*(u + u0)
    F0 = (rho/dt)*inner(u - u0, v)*dx + mu*inner(grad(U_cn), grad(v))*dx\
        + inner(grad(p0), v)*dx - inner(rho*f, v)*dx
    a0, L0 = system(F0)

    # Form for corrected pressure, solve to P
    U = Function(V)
    F1 = inner(grad(p - p0), grad(q))*dx + (rho/dt)*q*div(U)*dx
    a1, L1 = system(F1)

    # Form for corrected velocity
    P = Function(Q)
    F2 = (rho/dt)*inner(u - U, v)*dx + inner(grad(P - p0), v)*dx
    a2, L2 = system(F2)

    # Bcs for prediction and projection step
    bc_inlet = DirichletBC(V, u_inlet, boundaries, inlet_bdry)
    bc_wall = DirichletBC(V, u_wall, boundaries, wall_bdry)
    bcs_u = [bc_inlet, bc_wall]

    # Bcs for pressure correction
    bc_p = DirichletBC(Q, Constant(0.), boundaries, outlet_bdry)

    comm = mesh.mpi_comm()
    u_file = XDMFFile(comm, 'u_stokes_ipcs_{0}.xdmf'.format(mesh_))
    p_file = XDMFFile(comm, 'p_stokes_ipcs_{0}.xdmf'.format(mesh_))
    u_file.parameters['rewrite_function_mesh'] = False
    p_file.parameters['rewrite_function_mesh'] = False

    t, step = 0, 0
    while t < T_final:
        t += dt(0)
        step += 1
        print 'step {0}, time {1}'.format(step, t)

        # Update inlet
        u_inlet.t = t

        # Solve predictor step
        solve(a0 == L0, U, bcs_u)

        # Solve pressure correction step
        solve(a1 == L1, P, bc_p)

        # Solve velocity correction step
        solve(a2 == L2, u0, bcs_u)
        p0.assign(P)

        # Store
        if step % 10 == 0:
            plot(u0, title='Velocity @ t = {0}'.format(t))
            plot(P, title='Pressure @ t = {0}'.format(t))

            u_file << u0
            p_file << P

    interactive()


def yanenko_solver(mesh_):
    'Segregated solver using Yananko splitting. [Really like Chorin]'
    mesh = Mesh('../../meshes/%s.xml' % mesh_)
    boundaries =\
        MeshFunction('size_t', mesh, '../../meshes/%s_facet_region.xml' % mesh_)

    # These are all marked boundaries
    inlet_bdry = 14
    wall_bdry = 13  # includes cylinder
    outlet_bdry = 15

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    p = TrialFunction(Q)
    q = TestFunction(Q)

    dt = Constant(0.75*mesh.hmin()/U_m)

    # Form for predicted velocity, solve to U
    u0 = Function(V)
    U_cn = 0.5*(u + u0)
    F0 = (rho/dt)*inner(u - u0, v)*dx + mu*inner(grad(U_cn), grad(v))*dx\
        - inner(rho*f, v)*dx
    a0, L0 = system(F0)

    # Form for corrected pressure, solve to P
    U = Function(V)
    F1 = inner(grad(p), grad(q))*dx + (rho/dt)*q*div(U)*dx
    a1, L1 = system(F1)

    # Form for corrected velocity
    P = Function(Q)
    F2 = (rho/dt)*inner(u - U, v)*dx + inner(grad(P), v)*dx
    a2, L2 = system(F2)

    # Bcs for prediction and projection step
    bc_inlet = DirichletBC(V, u_inlet, boundaries, inlet_bdry)
    bc_wall = DirichletBC(V, u_wall, boundaries, wall_bdry)
    bcs_u = [bc_inlet, bc_wall]

    # Bcs for pressure correction
    bc_p = DirichletBC(Q, Constant(0.), boundaries, outlet_bdry)

    comm = mesh.mpi_comm()
    u_file = XDMFFile(comm, 'u_stokes_yank_{0}.xdmf'.format(mesh_))
    p_file = XDMFFile(comm, 'p_stokes_yank_{0}.xdmf'.format(mesh_))
    u_file.parameters['rewrite_function_mesh'] = False
    p_file.parameters['rewrite_function_mesh'] = False

    t, step = 0, 0
    while t < T_final:
        t += dt(0)
        step += 1
        print 'step {0}, time {1}'.format(step, t)

        # Update inlet
        u_inlet.t = t

        # Solve predictor step
        solve(a0 == L0, U, bcs_u)

        # Solve pressure correction step
        solve(a1 == L1, P, bc_p)

        # Solve velocity correction step
        solve(a2 == L2, u0, bcs_u)

        # Store
        if step % 10 == 0:
            plot(u0, title='Velocity @ t = {0}'.format(t))
            plot(P, title='Pressure @ t = {0}'.format(t))

            u_file << u0
            p_file << P

    interactive()
# -----------------------------------------------------------------------------

import sys

if sys.argv[1] == 'mixed':
    mixed_solver('turek_1')

elif sys.argv[1] == 'ipcs':
    ipcs_solver('turek_1')

elif sys.argv[1] == 'yank':
    yanenko_solver('turek_1')
