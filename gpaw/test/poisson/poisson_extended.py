import numpy as np

from gpaw.poisson import FDPoissonSolver as PoissonSolver
from gpaw.poisson_extended import ExtendedPoissonSolver
from gpaw.grid_descriptor import GridDescriptor

from gpaw.test import equal

# def equal(x, y, tol=0):
#     print '%.10e vs %.10e at %.10e' % (x, y, tol)


# Model grid
N_c = (16, 16, 3 * 16)
cell_cv = (1, 1, 3)
gd = GridDescriptor(N_c, cell_cv, False)

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

poissoneps = 1e-20

do_plot = False

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
            plt.sca(ax_i[ploti])
            ploti += 1
            plt.pcolormesh(big_phi_g[Ng_c[0] / 2])
            plt.sca(ax_i[ploti])
            ploti += 1
            plt.plot(big_phi_g[Ng_c[0] / 2, Ng_c[1] / 2])
            plt.ylim(np.array([-1, 1]) * 0.15)


def poisson_solve(gd, rho_g, poisson):
    phi_g = gd.zeros()
    npoisson = poisson.solve(phi_g, rho_g)
    return phi_g, npoisson


def compare(phi1_g, phi2_g, val):
    big_phi1_g = gd.collect(phi1_g)
    big_phi2_g = gd.collect(phi2_g)
    if gd.comm.rank == 0:
        equal(np.max(np.absolute(big_phi1_g - big_phi2_g)),
              val, np.sqrt(poissoneps))


# Get reference from default poissonsolver
poisson = PoissonSolver(eps=poissoneps)
poisson.set_grid_descriptor(gd)
phiref_g, npoisson = poisson_solve(gd, rho_g, poisson)

# Test agreement with default
poisson = ExtendedPoissonSolver(eps=poissoneps)
poisson.set_grid_descriptor(gd)
phi_g, npoisson = poisson_solve(gd, rho_g, poisson)
plot_phi(phi_g)
compare(phi_g, phiref_g, 0.0)

# Test moment_corrections=int
poisson = ExtendedPoissonSolver(eps=poissoneps, moment_corrections=4)
poisson.set_grid_descriptor(gd)
phi_g, npoisson = poisson_solve(gd, rho_g, poisson)
plot_phi(phi_g)
compare(phi_g, phiref_g, 4.1182101206e-02)

# Test moment_corrections=list
poisson = ExtendedPoissonSolver(eps=poissoneps,
    moment_corrections=[{'moms': range(4), 'center': [.5, .5, 1]},
                        {'moms': range(4), 'center': [.5, .5, 2]}])
poisson.set_grid_descriptor(gd)
phi_g, npoisson = poisson_solve(gd, rho_g, poisson)
plot_phi(phi_g)
compare(phi_g, phiref_g, 2.7569628594e-02)

# Test extendedgpts
poisson = ExtendedPoissonSolver(eps=poissoneps,
                                extended={'gpts': (24, 24, 3 * 24),
                                          'useprev': False})
poisson.set_grid_descriptor(gd)
phi_g, npoisson = poisson_solve(gd, rho_g, poisson)
plot_phi(phi_g)
compare(phi_g, phiref_g, 2.5351851105e-02)

# Test extendedgpts + moment_corrections, extendedhistory=False
poisson = ExtendedPoissonSolver(eps=poissoneps,
    extended={'gpts': (24, 24, 3 * 24),
              'useprev': False},
    moment_corrections=[{'moms': range(4), 'center': [.5, .5, 1]},
                        {'moms': range(4), 'center': [.5, .5, 2]}])
poisson.set_grid_descriptor(gd)
phi_g, npoisson = poisson_solve(gd, rho_g, poisson)
plot_phi(phi_g)
phi2_g, npoisson2 = poisson_solve(gd, rho_g, poisson)
equal(npoisson, npoisson2)
compare(phi_g, phi2_g, 0.0)
compare(phi_g, phiref_g, 2.75205170866e-02)

# Test extendedgpts + moment_corrections,
# extendedhistory=True
poisson = ExtendedPoissonSolver(eps=poissoneps,
    extended={'gpts': (24, 24, 3 * 24),
              'useprev': True},
    moment_corrections=[{'moms': range(4), 'center': [.5, .5, 1]},
                        {'moms': range(4), 'center': [.5, .5, 2]}])
poisson.set_grid_descriptor(gd)
phi_g, npoisson = poisson_solve(gd, rho_g, poisson)
phi2_g, npoisson2 = poisson_solve(gd, rho_g, poisson)
# The second run should use the old value -> niter=1
equal(npoisson2, 1)
# There is a slight difference in values
# because one more iteration in phi2_g
compare(phi_g, phi2_g, 0.0)


if do_plot:
    if gd.comm.rank == 0:
        plt.show()
