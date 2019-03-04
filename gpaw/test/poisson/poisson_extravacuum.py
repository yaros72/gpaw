from __future__ import print_function
import time
import numpy as np

from gpaw.poisson import PoissonSolver
from gpaw.poisson_extended import ExtendedPoissonSolver
from gpaw.poisson_extravacuum import ExtraVacuumPoissonSolver
from gpaw.grid_descriptor import GridDescriptor


do_output = False
do_plot = False
poissoneps = 1e-16

if do_output:
    def equal(x, y, tol=0):
        res = {True: 'ok', False: 'not ok'}[abs(x - y) < tol]
        print('%.10e vs %.10e at %.10e is %s' % (x, y, tol, res))
else:
    from gpaw.test import equal


# Model grid
N = 16
N_c = np.array((1, 1, 3)) * N
cell_c = N_c / float(N)
gd = GridDescriptor(N_c, cell_c, False)

# Construct model density
coord_vg = gd.get_grid_point_coordinates()
x_g = coord_vg[0, :]
y_g = coord_vg[1, :]
z_g = coord_vg[2, :]
rho_g = gd.zeros()
for z0 in [1, 2]:
    rho_g += 10 * (z_g - z0) * \
        np.exp(-20 * np.sum((coord_vg.T - np.array([.5, .5, z0])).T**2,
                            axis=0))

if do_plot:
    big_rho_g = gd.collect(rho_g)
    if gd.comm.rank == 0:
        import matplotlib.pyplot as plt
        fig, ax_ij = plt.subplots(3, 4, figsize=(20, 10))
        ax_i = ax_ij.ravel()
        ploti = 0
        Ng_c = gd.get_size_of_global_array()
        plt.sca(ax_i[ploti])
        ploti += 1
        plt.pcolormesh(big_rho_g[Ng_c[0] / 2])
        plt.sca(ax_i[ploti])
        ploti += 1
        plt.plot(big_rho_g[Ng_c[0] / 2, Ng_c[1] / 2])


def plot_phi(phi_g):
    if do_plot:
        big_phi_g = gd.collect(phi_g)
        if gd.comm.rank == 0:
            global ploti
            if ploti == 4:
                ploti -= 2
            plt.sca(ax_i[ploti])
            ploti += 1
            plt.pcolormesh(big_phi_g[Ng_c[0] / 2])
            plt.sca(ax_i[ploti])
            ploti += 1
            plt.plot(big_phi_g[Ng_c[0] / 2, Ng_c[1] / 2])
            plt.ylim(np.array([-1, 1]) * 0.15)


def poisson_solve(gd, rho_g, poisson):
    phi_g = gd.zeros()
    rho_g = rho_g
    t0 = time.time()
    npoisson = poisson.solve(phi_g, rho_g)
    t1 = time.time()
    if do_output:
        if gd.comm.rank == 0:
            print('Iterations: %s, Time: %.3f s' % (str(npoisson), t1 - t0))
    return phi_g, npoisson


def poisson_init_solve(gd, rho_g, poisson):
    poisson.set_grid_descriptor(gd)
    phi_g, npoisson = poisson_solve(gd, rho_g, poisson)
    plot_phi(phi_g)
    return phi_g, npoisson


def compare(phi1_g, phi2_g, val, eps=np.sqrt(poissoneps)):
    big_phi1_g = gd.collect(phi1_g)
    big_phi2_g = gd.collect(phi2_g)
    if gd.comm.rank == 0:
        diff = np.max(np.absolute(big_phi1_g - big_phi2_g))
    else:
        diff = 0.0
    diff = np.array([diff])
    gd.comm.broadcast(diff, 0)
    equal(diff[0], val, eps)


# Get reference from default poissonsolver
poisson = PoissonSolver('fd', eps=poissoneps)
phiref_g, npoisson = poisson_init_solve(gd, rho_g, poisson)

# Test agreement with default
poisson = ExtraVacuumPoissonSolver(N_c, PoissonSolver('fd', eps=poissoneps))
phi_g, npoisson = poisson_init_solve(gd, rho_g, poisson)
compare(phi_g, phiref_g, 0.0, 1e-24)

# New reference with extra vacuum
gpts = N_c * 4
poisson = ExtraVacuumPoissonSolver(gpts, PoissonSolver('fd', eps=poissoneps))
phi_g, npoisson = poisson_init_solve(gd, rho_g, poisson)
# print poisson.get_description()
compare(phi_g, phiref_g, 2.6485385144e-02)
phiref_g = phi_g

# Compare to ExtendedPoissonSolver
# TODO: remove this test when/if ExtendedPoissonSolver is deprecated
poisson = ExtendedPoissonSolver(eps=poissoneps,
                                extended={'gpts': gpts / 2,
                                          'useprev': True})
phi_g, npoisson = poisson_init_solve(gd, rho_g, poisson)
compare(phi_g, phiref_g, 0.0, 1e-24)

# Test with single coarsening
poisson = ExtraVacuumPoissonSolver(gpts, PoissonSolver('fd', eps=poissoneps),
                                   PoissonSolver('fd', eps=poissoneps), 1)
phi_g, npoisson = poisson_init_solve(gd, rho_g, poisson)
compare(phi_g, phiref_g, 1.5043946611e-04)

# Test with two coarsenings
poisson = ExtraVacuumPoissonSolver(gpts, PoissonSolver('fd', eps=poissoneps),
                                   PoissonSolver('fd', eps=poissoneps), 2)
phi_g, npoisson = poisson_init_solve(gd, rho_g, poisson)
compare(phi_g, phiref_g, 1.2980906205e-03)

# Test with cascaded single coarsenings
poisson1 = ExtraVacuumPoissonSolver(gpts / 2, PoissonSolver('fd', eps=poissoneps),
                                    PoissonSolver('fd', eps=poissoneps), 1)
poisson = ExtraVacuumPoissonSolver(gpts / 2, poisson1,
                                   PoissonSolver('fd', eps=poissoneps), 1)
phi_g, npoisson = poisson_init_solve(gd, rho_g, poisson)
# print poisson.get_description()
compare(phi_g, phiref_g, 1.7086531461e-04)

# Test auxgrid
gpts = N_c * 4
for coarses in [1, 2, 3]:
    for nn_refine in [1, 3]:
        for nn_laplace in [1, 3]:
            EVPS = ExtraVacuumPoissonSolver
            poisson = EVPS(gpts, PoissonSolver('fd', eps=poissoneps),
                           PoissonSolver('fd', eps=poissoneps), coarses,
                           use_aux_grid=False,
                           nn_refine=nn_refine, nn_laplace=nn_laplace)
            phiref_g, npoisson = poisson_init_solve(gd, rho_g, poisson)

            poisson = EVPS(gpts, PoissonSolver('fd', eps=poissoneps),
                           PoissonSolver('fd', eps=poissoneps), coarses,
                           use_aux_grid=True,
                           nn_refine=nn_refine, nn_laplace=nn_laplace)
            phi_g, npoisson = poisson_init_solve(gd, rho_g, poisson)
            compare(phi_g, phiref_g, 0.0, 1e-24)

if do_plot:
    if gd.comm.rank == 0:
        plt.show()
