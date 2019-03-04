import numpy as np


class XCFunctional(object):
    orbital_dependent = False

    def __init__(self, name, type):
        self.name = name
        self.gd = None
        self.ekin = 0.0
        self.type = type

    def todict(self):
        """Get dictionary representation of XC functional.

        This representation works for libxc kernels; other classes should
        likely override this function and should probably not rely on
        this implementation."""
        return {'type': self.kernel.type,
                'kernel': self.kernel.name}

    def tostring(self):
        """Get string representation of XC functional.

        This will give the name for libxc functionals but other data for
        hybrids."""
        return self.name

    def get_setup_name(self):
        return self.name

    def initialize(self, density, hamiltonian, wfs, occupations):
        pass

    def set_grid_descriptor(self, gd):
        self.gd = gd

    def calculate(self, gd, n_sg, v_sg=None, e_g=None):
        """Calculate energy and potential.

        gd: GridDescriptor
            Descriptor for 3-d grid.
        n_sg: rank-4 ndarray
            Spin densities.
        v_sg: rank-4 ndarray
            Array for potential.  The XC potential is added to the values
            already there.
        e_g: rank-3 ndarray
            Energy density.  Values must be written directly, not added.

        The total XC energy is returned."""

        if gd is not self.gd:
            self.set_grid_descriptor(gd)
        if e_g is None:
            e_g = gd.empty()
        if v_sg is None:
            v_sg = np.zeros_like(n_sg)
        self.calculate_impl(gd, n_sg, v_sg, e_g)
        return gd.integrate(e_g)

    def calculate_impl(self, gd, n_sg, v_sg, e_g):
        raise NotImplementedError

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None, a=None):
        raise NotImplementedError

    def set_positions(self, spos_ac, atom_partition=None):
        pass

    def get_description(self):
        """Get long description of functional as a string, or None."""
        return None

    def summary(self, fd):
        """Write summary of last calculation to file."""
        pass

    def write(self, writer, natoms=None):
        pass

    def read(self, reader):
        pass

    def estimate_memory(self, mem):
        pass

    # Orbital dependent stuff:
    def apply_orbital_dependent_hamiltonian(self, kpt, psit_nG,
                                            Htpsit_nG, dH_asp=None):
        pass

    def correct_hamiltonian_matrix(self, kpt, H_nn):
        # In what sense?  Some documentation here maybe?
        pass

    def add_correction(self, kpt, psit_xG, R_xG, P_axi, c_axi, n_x=None,
                       calculate_change=False):
        # Which kind of correction is this?  Maybe some kind of documentation
        # could be written?  What is required of an implementation?
        pass

    def rotate(self, kpt, U_nn):
        pass

    def get_kinetic_energy_correction(self):
        return self.ekin

    def add_forces(self, F_av):
        pass

    def calculate_spherical(self, rgd, n_sg, v_sg, e_g=None):
        raise NotImplementedError
