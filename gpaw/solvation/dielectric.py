from gpaw.solvation.gridmem import NeedsGD
import numpy as np


class Dielectric(NeedsGD):
    """Class representing a spatially varying permittivity.

    Attributes:
    eps_gradeps     -- List [eps_g, dxeps_g, dyeps_g, dzeps_g]
                       with
                       eps_g:   permittivity on the fine grid
                       dieps_g: gradient of eps_g.
    del_eps_del_g_g -- Partial derivative with respect to
                       the cavity. (A point-wise dependence
                       on the cavity is assumed).
    """

    def __init__(self, epsinf):
        """Constructor for the Dielectric class.

        Arguments:
        epsinf -- Static dielectric constant
                  at infinite distance from the solute.
        """
        NeedsGD.__init__(self)
        self._epsinf = float(epsinf)
        self.eps_gradeps = None  # eps_g, dxeps_g, dyeps_g, dzeps_g
        self.del_eps_del_g_g = None
        self.cavity = None

    def estimate_memory(self, mem):
        nbytes = self.gd.bytecount()
        mem.subnode('Permittivity', nbytes)
        mem.subnode('Permittivity Gradient', 3 * nbytes)
        mem.subnode('Permittivity Derivative', nbytes)

    def allocate(self):
        NeedsGD.allocate(self)
        self.eps_gradeps = []
        eps_g = self.gd.empty()
        eps_g.fill(1.0)
        self.eps_gradeps.append(eps_g)
        self.eps_gradeps.extend([gd.zeros() for gd in (self.gd, ) * 3])
        self.del_eps_del_g_g = self.gd.empty()

    @property
    def epsinf(self):
        return self._epsinf

    @epsinf.setter
    def epsinf(self, epsinf):
        if epsinf != self._epsinf:
            self._epsinf = float(epsinf)
            self.del_eps_del_g_g = self._epsinf - 1.
            if self.cavity is not None:
                self.update(self.cavity)

    def update(self, cavity):
        """Calculate eps_gradeps and del_eps_del_g_g from the cavity."""
        self.cavity = cavity
        self.update_eps_only()
        for i in (0, 1, 2):
            np.multiply(
                self.del_eps_del_g_g,
                cavity.grad_g_vg[i],
                self.eps_gradeps[1 + i]
            )

    def update_eps_only(self):
        raise NotImplementedError

    def __str__(self):
        s = 'Dielectric: %s\n' % (self.__class__, )
        s += '  epsilon_inf: %s\n' % (self._epsinf, )
        return s


class LinearDielectric(Dielectric):
    """Dielectric depending (affine) linearly on the cavity.

    See also
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).
    """
    def allocate(self):
        Dielectric.allocate(self)
        self.del_eps_del_g_g = self._epsinf - 1.  # frees array

    def update_eps_only(self):
        np.multiply(self.cavity.g_g, self._epsinf - 1., self.eps_gradeps[0])
        self.eps_gradeps[0] += 1.


class CMDielectric(Dielectric):
    """Clausius-Mossotti like dielectric.

    Untested, use at own risk!
    """

    def update_eps_only(self):
        ei = self._epsinf
        t = 1. - self.cavity.g_g
        self.eps_gradeps[0][:] = (3. * (ei + 2.)) / ((ei - 1.) * t + 3.) - 2.
        self.del_eps_del_g_g[:] = (
            (3. * (ei - 1.) * (ei + 2.)) / ((ei - 1.) * t + 3.) ** 2
        )
