"""This module defines different external potentials."""
import warnings

import numpy as np

from ase.units import Bohr, Hartree

import _gpaw

__all__ = ['ConstantPotential', 'ConstantElectricField', 'CDFTPotential']


def create_external_potential(name, **kwargs):
    """Construct potential from dict."""
    if name not in __all__:
        raise ValueError
    if name == 'CDFTPotential':
        return None
    return globals()[name](**kwargs)


class ExternalPotential:
    vext_g = None
    vext_q = None

    def get_potential(self, gd):
        """Get the potential on a regular 3-d grid.

        Will only call calculate_potential() the first time."""

        if self.vext_g is None:
            self.calculate_potential(gd)
            self.vext_g.flags.writeable = False
        return self.vext_g

    def get_potentialq(self, gd, pd3):
        """Get the potential on a regular 3-d grid in real space.

        Will only call calculate_potential() the first time."""

        if self.vext_q is None:
            vext_g = self.get_potential(gd)
            self.vext_q = pd3.fft(vext_g)
            self.vext_q.flags.writeable = False

        return self.vext_q

    def calculate_potential(self, gd):
        raise NotImplementedError

    def get_name(self):
        return self.__class__.__name__


class ConstantPotential(ExternalPotential):
    """Constant potential for tests."""
    def __init__(self, constant=1.0):
        self.constant = constant / Hartree
        self.name = 'ConstantPotential'

    def __str__(self):
        return 'Constant potential: {:.3f} V'.format(self.constant * Hartree)

    def calculate_potential(self, gd):
        self.vext_g = gd.zeros() + self.constant

    def todict(self):
        return {'name': self.name,
                'constant': self.constant * Hartree}


class ConstantElectricField(ExternalPotential):
    def __init__(self, strength, direction=[0, 0, 1], tolerance=1e-7):
        """External constant electric field.

        strength: float
            Field strength in V/Ang.
        direction: vector
            Polarisation direction.
        """
        d_v = np.asarray(direction)
        self.field_v = strength * d_v / (d_v**2).sum()**0.5 * Bohr / Hartree
        self.tolerance = tolerance
        self.name = 'ConstantElectricField'

    def __str__(self):
        return ('Constant electric field: '
                '({:.3f}, {:.3f}, {:.3f}) V/Ang'
                .format(*(self.field_v * Hartree / Bohr)))

    def calculate_potential(self, gd):
        # Currently skipped, PW mode is periodic in all directions
        # d_v = self.field_v / (self.field_v**2).sum()**0.5
        # for axis_v in gd.cell_cv[gd.pbc_c]:
        #     if abs(np.dot(d_v, axis_v)) > self.tolerance:
        #         raise ValueError(
        #             'Field not perpendicular to periodic axis: {}'
        #             .format(axis_v))

        center_v = 0.5 * gd.cell_cv.sum(0)
        r_gv = gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        self.vext_g = np.dot(r_gv - center_v, self.field_v)

    def todict(self):
        strength = (self.field_v**2).sum()**0.5
        return {'name': self.name,
                'strength': strength * Hartree / Bohr,
                'direction': self.field_v / strength}


class PointChargePotential(ExternalPotential):
    def __init__(self, charges, positions=None,
                 rc=0.2, rc2=np.inf, width=1.0):
        """Point-charge potential.

        charges: list of float
            Charges.
        positions: (N, 3)-shaped array-like of float
            Positions of charges in Angstrom.  Can be set later.
        rc: float
            Inner cutoff for Coulomb potential in Angstrom.
        rc2: float
            Outer cutoff for Coulomb potential in Angstrom.
        width: float
            Width for cutoff function for Coulomb part.

        For r < rc, 1 / r is replaced by a third order polynomial in r^2 that
        has matching value, first derivative, second derivative and integral.

        For rc2 - width < r < rc2, 1 / r is multiplied by a smooth cutoff
        function (a third order polynomium in r).

        You can also give rc a negative value.  In that case, this formula
        is used::

            (r^4 - rc^4) / (r^5 - |rc|^5)

        for all values of r - no cutoff at rc2!
        """
        self.q_p = np.ascontiguousarray(charges, float)
        self.rc = rc / Bohr
        self.rc2 = rc2 / Bohr
        self.width = width / Bohr
        if positions is not None:
            self.set_positions(positions)
        else:
            self.R_pv = None

        if abs(self.q_p).max() < 1e-14:
            warnings.warn('No charges!')
        if self.rc < 0. and self.rc2 < np.inf:
            warnings.warn('Long range cutoff chosen but will not be applied\
                           for negative inner cutoff values!')

    def __str__(self):
        return ('Point-charge potential '
                '(points: {}, cutoffs: {:.3f}, {:.3f}, {:.3f} Ang)'
                .format(len(self.q_p),
                        self.rc * Bohr,
                        (self.rc2 - self.width) * Bohr,
                        self.rc2 * Bohr))

    def set_positions(self, R_pv, com_pv=None):
        """Update positions."""
        if com_pv is not None:
            self.com_pv = np.asarray(com_pv) / Bohr
        else:
            self.com_pv = None

        self.R_pv = np.asarray(R_pv) / Bohr
        self.vext_g = None

    def _molecule_distances(self, gd):
        if self.com_pv is not None:
            return self.com_pv - gd.cell_cv.sum(0) / 2

    def calculate_potential(self, gd):
        assert gd.orthogonal
        self.vext_g = gd.zeros()

        dcom_pv = self._molecule_distances(gd)

        _gpaw.pc_potential(gd.beg_c, gd.h_cv.diagonal().copy(),
                           self.q_p, self.R_pv,
                           self.rc, self.rc2, self.width,
                           self.vext_g, dcom_pv)

    def get_forces(self, calc):
        """Calculate forces from QM charge density on point-charges."""
        dens = calc.density
        F_pv = np.zeros_like(self.R_pv)
        gd = dens.finegd
        dcom_pv = self._molecule_distances(gd)

        _gpaw.pc_potential(gd.beg_c, gd.h_cv.diagonal().copy(),
                           self.q_p, self.R_pv,
                           self.rc, self.rc2, self.width,
                           self.vext_g, dcom_pv, dens.rhot_g, F_pv)
        gd.comm.sum(F_pv)
        return F_pv * Hartree / Bohr

class CDFTPotential(ExternalPotential):
    # Dummy class to make cDFT compatible with new external potential class ClassName(object):
    def __init__(self):
        self.name = 'CDFTPotential'
