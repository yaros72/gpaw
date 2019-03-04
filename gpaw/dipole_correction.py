import numpy as np
from ase.units import Bohr
from ase.utils import basestring

from gpaw.utilities import erf


class DipoleCorrection:
    """Dipole-correcting wrapper around another PoissonSolver."""
    def __init__(self, poissonsolver, direction, width=1.0):
        """Construct dipole correction object.

        poissonsolver:
            Poisson solver.
        direction: int or str
            Specification of layer: 0, 1, 2, 'xy', 'xz' or 'yz'.
        width: float
            Width in Angstrom of dipole layer used for the plane-wave
            implementation.
        """
        self.c = direction
        self.poissonsolver = poissonsolver
        self.width = width / Bohr

        self.correction = None  # shift in potential
        self.sawtooth_q = None  # Fourier transformed sawtooth

    def todict(self):
        dct = self.poissonsolver.todict()
        dct['dipolelayer'] = self.c
        if self.width != 1.0 / Bohr:
            dct['width'] = self.width * Bohr
        return dct

    def get_stencil(self):
        return self.poissonsolver.get_stencil()

    def set_grid_descriptor(self, gd):
        self.poissonsolver.set_grid_descriptor(gd)
        self.check_direction(gd, gd.pbc_c)

    def check_direction(self, gd, pbc_c):
        if isinstance(self.c, basestring):
            axes = ['xyz'.index(d) for d in self.c]
            for c in range(3):
                if abs(gd.cell_cv[c, axes]).max() < 1e-12:
                    break
            else:
                raise ValueError('No axis perpendicular to {0}-plane!'
                                 .format(self.c))
            self.c = c

        if pbc_c[self.c]:
            raise ValueError('System must be non-periodic perpendicular '
                             'to dipole-layer.')

        # Right now the dipole correction must be along one coordinate
        # axis and orthogonal to the two others.  The two others need not
        # be orthogonal to each other.
        for c1 in range(3):
            if c1 != self.c:
                if abs(np.dot(gd.cell_cv[self.c], gd.cell_cv[c1])) > 1e-12:
                    raise ValueError('Dipole correction axis must be '
                                     'orthogonal to the two other axes.')

    def get_description(self):
        poissondesc = self.poissonsolver.get_description()
        desc = 'Dipole correction along %s-axis' % 'xyz'[self.c]
        return '\n'.join([poissondesc, desc])

    def initialize(self):
        self.poissonsolver.initialize()

    def solve(self, pot, dens, **kwargs):
        if isinstance(dens, np.ndarray):
            # finite-diference Poisson solver:
            return self.fdsolve(pot, dens, **kwargs)
        # Plane-wave solver:
        self.pwsolve(pot, dens)

    def fdsolve(self, vHt_g, rhot_g, **kwargs):
        gd = self.poissonsolver.gd
        drhot_g, dvHt_g, self.correction = dipole_correction(
            self.c, gd, rhot_g)
        vHt_g -= dvHt_g
        iters = self.poissonsolver.solve(vHt_g, rhot_g + drhot_g, **kwargs)
        vHt_g += dvHt_g
        return iters

    def pwsolve(self, vHt_q, dens):
        gd = self.poissonsolver.pd.gd

        if self.sawtooth_q is None:
            self.initialize_sawtooth()

        self.poissonsolver.solve(vHt_q, dens)
        dip_v = dens.calculate_dipole_moment()
        c = self.c
        L = gd.cell_cv[c, c]
        self.correction = 2 * np.pi * dip_v[c] * L / gd.volume
        vHt_q -= 2 * self.correction * self.sawtooth_q

    def initialize_sawtooth(self):
        gd = self.poissonsolver.pd.gd
        self.check_direction(gd, self.poissonsolver.realpbc_c)
        if gd.comm.rank == 0:
            c = self.c
            L = gd.cell_cv[c, c]
            w = self.width / 2
            assert w < L / 2
            gc = int(w / gd.h_cv[c, c])
            x = gd.coords(c)
            sawtooth = x / L - 0.5
            a = 1 / L - 0.75 / w
            b = 0.25 / w**3
            sawtooth[:gc] = x[:gc] * (a + b * x[:gc]**2)
            sawtooth[-gc:] = -sawtooth[gc:0:-1]
            sawtooth_g = gd.empty(global_array=True)
            shape = [1, 1, 1]
            shape[c] = -1
            sawtooth_g[:] = sawtooth.reshape(shape)
            sawtooth_q = self.poissonsolver.pd.fft(sawtooth_g, local=True)
        else:
            sawtooth_q = None
        self.sawtooth_q = self.poissonsolver.pd.scatter(sawtooth_q)

    def estimate_memory(self, mem):
        self.poissonsolver.estimate_memory(mem)


def dipole_correction(c, gd, rhot_g):
    """Get dipole corrections to charge and potential.

    Returns arrays drhot_g and dphit_g such that if rhot_g has the
    potential phit_g, then rhot_g + drhot_g has the potential
    phit_g + dphit_g, where dphit_g is an error function.

    The error function is chosen so as to be largely constant at the
    cell boundaries and beyond.
    """
    # This implementation is not particularly economical memory-wise

    moment = gd.calculate_dipole_moment(rhot_g)[c]
    if abs(moment) < 1e-12:
        return gd.zeros(), gd.zeros(), 0.0

    r_g = gd.get_grid_point_coordinates()[c]
    cellsize = abs(gd.cell_cv[c, c])
    sr_g = 2.0 / cellsize * r_g - 1.0  # sr ~ 'scaled r'
    alpha = 12.0  # should perhaps be variable
    drho_g = sr_g * np.exp(-alpha * sr_g**2)
    moment2 = gd.calculate_dipole_moment(drho_g)[c]
    factor = -moment / moment2
    drho_g *= factor
    phifactor = factor * (np.pi / alpha)**1.5 * cellsize**2 / 4.0
    dphi_g = -phifactor * erf(sr_g * np.sqrt(alpha))
    return drho_g, dphi_g, phifactor
