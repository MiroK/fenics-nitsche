'''
    Steady 2d benchmark from
        [Turek et al;
        Benchmark Computations of Laminar Flow Around a Cylinder]

    We usde Nietche formulation from
        [Becker;
        Mesh adaptation for Dirichlet flow control via Nitsche's method]
'''

from dolfin import *

U_m = 0.3
H = 0.41

f = Constant((0., 0.))
u_inlet = Expression(('4*U_m*x[1]*(H-x[1])/H/H', '0'), U_m=U_m, H=H)
u_wall = Constant((0., 0.))
mu = Constant(1E-3)


def nitsche(mesh_):
    mesh = Mesh('../meshes/%s.xml' % mesh_)
    boundaries =\
        MeshFunction('size_t', mesh, '../meshes/%s_facet_region.xml' % mesh_)

    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    M = MixedFunctionSpace([V, Q])

    u, p = TrialFunctions(M)
    v, q = TestFunctions(M)

    up0 = Function(M)
    u0, p0 = split(up0)

    # These are all marked boundaries
    inlet_bdry = 14
    wall_bdry = 13  # includes cylinder
    outlet_bdry = 15

    ds = Measure('ds')[boundaries]
    n = FacetNormal(mesh)
    h = mesh.ufl_cell().max_facet_edge_length

    # Define forms
    # Note that (skew)-symmetry is both in `standard`=dx terms
    # and also Nitsche ones.
    a = inner(dot(grad(u), u0), v)*dx + inner(mu*grad(u), grad(v))*dx\
        - inner(p, div(v))*dx - inner(q, div(u))*dx

    L = inner(f, v)*dx

    # Add velocity bcs weakly via symmetric Nitsche
    beta = Constant(100.)
    ds_u = ds(inlet_bdry) + ds(wall_bdry)
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
        - inner(dot(mu*grad(v), n), u_wall)*ds(wall_bdry)

    # Finaly we add terms that enforce only normal component of velocity
    # in case the flow is inviscid and terms that set bcs only on the inflow -
    # this is usefull for convection dominnt flows
    beta2 = Constant(100.)
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
    iter_max = 100
    converged = False
    tol = 1E-14
    e = -1
    while not converged and iter < iter_max:
        iter += 1
        # print 'Iteration', iter

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

        # print '\t Error', e

    u, p = up.split(True)

    # As I forgot to mark the cylinder, I only compute pressure difference
    delta_p = p(0.15, 0.2) - p(0.25, 0.2)

    return delta_p

# -----------------------------------------------------------------------------

from collections import namedtuple

Result = namedtuple('Result', ['delta_p', 'passed', 'error'])

results = []
for mesh in ['turek_1', 'turek_2', 'turek_3']:
    delta_p = nitsche(mesh_=mesh)
    e = min(abs(delta_p - 0.1172), abs(delta_p - 0.1176))
    results.append(Result(delta_p, between(delta_p, (0.1172, 0.1176)), e))

for result in results:
    print result
