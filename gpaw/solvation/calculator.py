from gpaw import GPAW
from gpaw.solvation.hamiltonian import SolvationRealSpaceHamiltonian
from ase.units import Hartree, Bohr


class SolvationGPAW(GPAW):
    """Subclass of gpaw.GPAW calculator with continuum solvent model.

    See also Section III of
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).
    """

    def __init__(self, cavity, dielectric, interactions=None,
                 **gpaw_kwargs):
        """Constructor for SolvationGPAW class.

        Additional arguments not present in GPAW class:
        cavity       -- A Cavity instance.
        dielectric   -- A Dielectric instance.
        interactions -- A list of Interaction instances.
        """
        if interactions is None:
            interactions = []

        self.stuff_for_hamiltonian = (cavity, dielectric, interactions)

        GPAW.__init__(self, **gpaw_kwargs)

    def create_hamiltonian(self, realspace, mode, xc):
        if not realspace:
            raise NotImplementedError(
                'SolvationGPAW does not support '
                'calculations in reciprocal space yet.')

        dens = self.density

        self.hamiltonian = SolvationRealSpaceHamiltonian(
            *self.stuff_for_hamiltonian,
            gd=dens.gd, finegd=dens.finegd,
            nspins=dens.nspins,
            collinear=dens.collinear,
            setups=dens.setups,
            timer=self.timer,
            xc=xc,
            world=self.world,
            redistributor=dens.redistributor,
            vext=self.parameters.external,
            psolver=self.parameters.poissonsolver,
            stencil=mode.interpolation)

        self.log(self.hamiltonian)

        xc.set_grid_descriptor(self.hamiltonian.finegd)

    def initialize_positions(self, atoms=None):
        spos_ac = GPAW.initialize_positions(self, atoms)
        self.hamiltonian.update_atoms(self.atoms, self.log)
        return spos_ac

    def get_electrostatic_energy(self):
        """Return electrostatic part of the total energy.

        The electrostatic part consists of everything except
        the short-range interactions defined in the interactions list.
        """
        # Energy extrapolated to zero width:
        return Hartree * self.hamiltonian.e_el_extrapolated

    def get_solvation_interaction_energy(self, subscript, atoms=None):
        """Return a specific part of the solvation interaction energy.

        The subscript parameter defines which part is to be returned.
        It has to match the value of a subscript attribute of one of
        the interactions in the interactions list.
        """
        #self.calculate(atoms, converge=True)
        return Hartree * getattr(self.hamiltonian, 'e_' + subscript)

    def get_cavity_volume(self, atoms=None):
        """Return the cavity volume in Angstrom ** 3.

        In case no volume calculator has been set for the cavity, None
        is returned.
        """
        #self.calculate(atoms, converge=True)
        V = self.hamiltonian.cavity.V
        return V and V * Bohr ** 3

    def get_cavity_surface(self, atoms=None):
        """Return the cavity surface area in Angstrom ** 2.

        In case no surface calculator has been set for the cavity,
        None is returned.
        """
        #self.calculate(atoms, converge=True)
        A = self.hamiltonian.cavity.A
        return A and A * Bohr ** 2

    def write(self, *args, **kwargs):
        raise NotImplementedError(
            'IO is not implemented yet for SolvationGPAW!')
