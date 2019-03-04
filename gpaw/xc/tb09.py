from __future__ import print_function
"""Tran-Blaha potential.

From:

    Accurate Band Gaps of Semiconductors and Insulators
    with a Semilocal Exchange-Correlation Potential

    Fabien Tran and Peter Blaha

    PRL 102, 226401 (2009)

    DOI: 10.1103/PhysRevLett.102.226401

"""
import numpy as np
from ase.utils import seterr

from gpaw.xc.mgga import MGGA, weight_n
from gpaw.xc.libxc import LibXC
from gpaw.fd_operators import Laplace


class TB09Kernel:
    name = 'TB09'
    type = 'MGGA'
    alpha = -0.012
    beta = 1.023

    def __init__(self, c=None):
        self.tb09 = LibXC('MGGA_X_TB09').xc.tb09
        self.ldac = LibXC('LDA_C_PW')

        self.fixedc = c is not None  # calculate c or use fixed value
        self.c = c  # amount of "exact exchange"
        self.n = 0  # Lebedev quadrature point number (0-49)
        self.sign = 1.0  # sign of PAW correction: +1 for AE and -1 for PS
        self.I = None  # integral from Eq. (3)

    def calculate(self, e_g, n_sg, dedn_sg, sigma_xg,
                  dedsigma_xg, tau_sg, dedtau_sg):
        ns = len(n_sg)
        n_sg[n_sg < 1e-6] = 1e-6

        if n_sg.ndim == 4:
            if not self.fixedc:
                if self.c is None:
                    # We don't have the integral yet - just use 1.0:
                    self.c = 1.0
                else:
                    self.I = self.world.sum(self.I)
                    self.c = (self.alpha + self.beta *
                              (self.I / self.gd.volume)**0.5)

                # Start calculation of c for use in the next SCF step:
                if ns == 1:
                    gradn_g = sigma_xg[0]**0.5
                else:
                    gradn_g = (sigma_xg[0] +
                               2 * sigma_xg[1] +
                               sigma_xg[2])**0.5
                self.I = self.gd.integrate(gradn_g / n_sg.sum(0))
                # The domain is not distributed like the PAW corrections:
                self.I /= self.world.size

            lapl_sg = self.gd.empty(ns)
            for n_g, lapl_g in zip(n_sg, lapl_sg):
                self.lapl.apply(n_g, lapl_g)

        else:
            rgd = self.rgd
            lapl_sg = []
            for n_Lg in self.n_sLg:
                lapl_g = rgd.laplace(np.dot(self.Y_L, n_Lg))
                l = 0
                L1 = 0
                while L1 < len(self.Y_L):
                    L2 = L1 + 2 * l + 1
                    n_g = np.dot(self.Y_L[L1:L2], n_Lg[L1:L2])
                    with seterr(divide='ignore', invalid='ignore'):
                        lapl_g -= l * (l + 1) * n_g / rgd.r_g**2
                    lapl_g[0] = 0.0
                    L1 = L2
                    l += 1
                lapl_sg.append(lapl_g)

            if not self.fixedc:
                # PAW corrections to integral:
                w = self.sign * weight_n[self.n]
                if ns == 1:
                    gradn_g = sigma_xg[0]**0.5
                else:
                    gradn_g = (sigma_xg[0] +
                               2 * sigma_xg[1] +
                               sigma_xg[2])**0.5
                self.I += w * rgd.integrate(gradn_g / n_sg.sum(0))

                self.n += 1
                if self.n == len(weight_n):
                    self.n = 0
                    self.sign = -self.sign

        # dedn_sg[:] = 0.0
        sigma_xg[sigma_xg < 1e-10] = 1e-10
        tau_sg[tau_sg < 1e-10] = 1e-10

        for n_g, sigma_g, lapl_g, tau_g, v_g in zip(n_sg, sigma_xg[::2],
                                                    lapl_sg, tau_sg, dedn_sg):
            self.tb09(self.c, n_g.ravel(), sigma_g, lapl_g, tau_g, v_g,
                      dedsigma_xg)

        self.ldac.calculate(e_g, n_sg, dedn_sg)
        e_g[:] = 0.0

        dedsigma_xg[:] = 0.0
        dedtau_sg[:] = 0.0


class TB09(MGGA):
    def __init__(self, c=None, stencil=2):
        MGGA.__init__(self, TB09Kernel(c), stencil=stencil)

    def get_setup_name(self):
        return 'LDA'

    def initialize(self, dens, ham, wfs, occ):
        MGGA.initialize(self, dens, ham, wfs, occ)
        self.kernel.world = wfs.world
        self.kernel.gd = dens.finegd
        self.kernel.lapl = Laplace(dens.finegd)

    def create_mgga_radial_calculator(self):
        rcalc = MGGA.create_mgga_radial_calculator(self)

        def f(rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv, n):
            self.kernel.n_sLg = n_sLg
            self.kernel.Y_L = Y_L
            self.kernel.rgd = rgd
            return rcalc(rgd, n_sLg, Y_L, dndr_sLg, rnablaY_Lv, n)

        return f

    def apply_orbital_dependent_hamiltonian(self, kpt, psit_xG,
                                            Htpsit_xG, dH_asp):
        pass

    @property
    def c(self):
        """Amount of "exact exchange"."""
        return self.kernel.c
