from gpaw.solvation.gridmem import NeedsGD
from ase.units import Bohr, Hartree
import numpy as np


class Interaction(NeedsGD):
    """Base class for non-electrostatic solvent solute interactions.

    See also Section III of
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).

    Attributes:
    subscript         -- Short label used to reference the interaction.
    E                 -- Local interaction energy in Hartree
    delta_E_delta_n_g -- Functional derivative of the interaction energy
                         with respect to the electron density.
    delta_E_delta_n_g -- Functional derivative of the interaction energy
                         with respect to the cavity.
    """

    subscript = 'unnamed'

    def __init__(self):
        NeedsGD.__init__(self)
        self.E = None
        self.delta_E_delta_n_g = None
        self.delta_E_delta_g_g = None

    def update(self, atoms, density, cavity):
        """Update the Kohn-Sham potential and the energy.

        atoms and/or cavity are None iff they have not changed
        since the last call

        Return whether the interaction has changed.
        """
        raise NotImplementedError

    def estimate_memory(self, mem):
        ngrids = 1 + self.depends_on_el_density
        mem.subnode('Functional Derivatives', ngrids * self.gd.bytecount())

    def allocate(self):
        NeedsGD.allocate(self)
        self.delta_E_delta_g_g = self.gd.empty()
        if self.depends_on_el_density:
            self.delta_E_delta_n_g = self.gd.empty()

    @property
    def depends_on_atomic_positions(self):
        """Return whether the ia depends explicitly on atomic positions."""
        raise NotImplementedError

    @property
    def depends_on_el_density(self):
        """Return whether the ia depends explicitly on the electron density."""
        raise NotImplementedError

    def get_del_r_vg(self, atomic_index, density):
        """Return spatial derivatives with respect to atomic position."""
        raise NotImplementedError

    def __str__(self):
        s = "Interaction: %s\n" % (self.__class__, )
        s += "  subscript: %s\n" % (self.subscript, )
        return s

    def update_atoms(self, atoms, log):
        """Inexpensive initialization when atoms change."""
        pass


class SurfaceInteraction(Interaction):
    """An interaction with energy proportional to the cavity surface area."""

    subscript = 'surf'
    depends_on_el_density = False
    depends_on_atomic_positions = False

    def __init__(self, surface_tension):
        """Constructor for SurfaceInteraction class.

        Arguments:
        surface_tension -- Proportionality factor to calculate
                           energy from surface area in eV / Angstrom ** 2.
        """
        Interaction.__init__(self)
        self.surface_tension = float(surface_tension)

    def update(self, atoms, density, cavity):
        if cavity is None:
            return False
        acalc = cavity.surface_calculator
        st = self.surface_tension * Bohr ** 2 / Hartree
        self.E = st * acalc.A
        np.multiply(st, acalc.delta_A_delta_g_g, self.delta_E_delta_g_g)
        return True

    def __str__(self):
        s = Interaction.__str__(self)
        s += '  surface_tension: %s\n' % (self.surface_tension, )
        return s


class VolumeInteraction(Interaction):
    """An interaction with energy proportional to the cavity volume"""

    subscript = 'vol'
    depends_on_el_density = False
    depends_on_atomic_positions = False

    def __init__(self, pressure):
        """Constructor for VolumeInteraction class.

        Arguments:
        pressure -- Proportionality factor to calculate
                    energy from volume in eV / Angstrom ** 3.
        """
        Interaction.__init__(self)
        self.pressure = float(pressure)

    def update(self, atoms, density, cavity):
        if cavity is None:
            return False
        vcalc = cavity.volume_calculator
        pressure = self.pressure * Bohr ** 3 / Hartree
        self.E = pressure * vcalc.V
        np.multiply(pressure, vcalc.delta_V_delta_g_g, self.delta_E_delta_g_g)
        return True

    def __str__(self):
        s = Interaction.__str__(self)
        s += '  pressure: %s\n' % (self.pressure, )
        return s


class LeakedDensityInteraction(Interaction):
    """Interaction proportional to charge leaking outside cavity.

    The charge outside the cavity is calculated as
    """

    subscript = 'leak'
    depends_on_el_density = True
    depends_on_atomic_positions = False

    def __init__(self, voltage):
        """Constructor for LeakedDensityInteraction class.

        Arguments:
        voltage -- Proportionality factor to calculate
                   energy from integrated electron density in Volts.
                   A positive value of the voltage leads to a
                   positive interaction energy.
        """
        Interaction.__init__(self)
        self.voltage = float(voltage)

    def update(self, atoms, density, cavity):
        E0 = self.voltage / Hartree
        if cavity is not None:
            np.multiply(E0, cavity.g_g, self.delta_E_delta_n_g)
        np.multiply(E0, density.nt_g, self.delta_E_delta_g_g)
        self.E = self.gd.integrate(
            density.nt_g * self.delta_E_delta_n_g,
            global_integral=False
        )
        return True

    def __str__(self):
        s = Interaction.__str__(self)
        s += '  voltage: %s\n' % (self.voltage, )
        return s
