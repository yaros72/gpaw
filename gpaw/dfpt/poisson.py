# Copyright (C) 2003  CAMP
# Please see the accompanying LICENSE file for further information.

# FIXME: This code used to be used in dfpt/phononcalculator.py in replacement
#        for the original FFTPoissonSolver. This code should be dead.

from math import pi

from numpy.fft import fftn, ifftn

from gpaw.utilities.tools import construct_reciprocal
from gpaw import poisson

class PoissonSolver(poisson.FDPoissonSolver):
    """Copy of GPAW Poissonsolver"""

class FFTPoissonSolver(poisson.FFTPoissonSolver):

    def __init__(self, eps=2e-10, dtype=float):
        """Set the ``dtype`` of the source term array."""

        self.dtype = dtype
        self.charged_periodic_correction = None
        self.eps = eps

    def set_q(self, q_c):
        """Set q-vector in case of Bloch-type charge distribution.

        Parameters
        ----------
        q_c: ndarray
            q-vector in scaled coordinates of the reciprocal lattice vectors.

        """

        if self.gd.comm.rank == 0:
            if self.gd.comm.size > 1:
                raise RuntimeError("This solver is obsolete:" +
                    "domain decomposition support has been removed")
            self.k2_Q, self.N3 = construct_reciprocal(self.gd, q_c=q_c)

    def solve_neutral(self, phi_g, rho_g, eps=None):
        """Solve Poissons equation for a neutral and periodic charge density.

        Parameters
        ----------
        phi_g: ndarray
            Potential (output array).
        rho_g: ndarray
            Charge distribution (in units of -e).

        """

        assert phi_g.dtype == self.dtype
        assert rho_g.dtype == self.dtype

        if self.gd.comm.size == 1:
            # Note, implicit downcast from complex to float when the dtype of
            # phi_g is float
            phi_g[:] = ifftn(fftn(rho_g) * 4.0 * pi / self.k2_Q)
        else:
            rho_g = self.gd.collect(rho_g)
            if self.gd.comm.rank == 0:
                globalphi_g = ifftn(fftn(rho_g) * 4.0 * pi / self.k2_Q)
            else:
                globalphi_g = None
            # What happens here if globalphi is complex and phi is real ??????
            self.gd.distribute(globalphi_g, phi_g)

        return 1
