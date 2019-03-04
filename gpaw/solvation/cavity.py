from ase.units import kB, Hartree, Bohr
from gpaw.solvation.gridmem import NeedsGD
from gpaw.fd_operators import Gradient
from gpaw.io.logger import indent
import numpy as np


BAD_RADIUS_MESSAGE = "All atomic radii have to be finite and >= zero."


def set_log_and_check_radii(obj, atoms, log):
    radii = np.array(obj.atomic_radii(atoms), dtype=float)
    obj.atomic_radii_output = radii
    obj.symbols = atoms.get_chemical_symbols()
    log('  Atomic radii for %s:' % (obj.__class__, ))
    for a, (s, r) in enumerate(
            zip(obj.symbols, radii)):
        log('    %3d %-2s %10.5f' % (a, s, r))
    if not np.isfinite(radii).all() or (radii < 0).any():
        raise ValueError(BAD_RADIUS_MESSAGE)


def get_pbc_positions(atoms, r_max):
    """Return dict mapping atom index to positions in Bohr.

    With periodic boundary conditions, it also includes neighbouring
    cells up to a distance of r_max (in Bohr).
    """
    # code snippet taken from ase/calculators/vdwcorrection.py
    pbc_c = atoms.get_pbc()
    cell_cv = atoms.get_cell() / Bohr
    Rcell_c = np.sqrt(np.sum(cell_cv ** 2, axis=1))
    ncells_c = np.ceil(np.where(pbc_c, 1. + r_max / Rcell_c, 1)).astype(int)
    pos_aav = {}
    # loop over all atoms in the cell (and neighbour cells for PBC)
    for index1, atom in enumerate(atoms):
        pos = atom.position / Bohr
        pos_aav[index1] = np.empty((np.prod(ncells_c * 2 - 1), 3))
        # loops over neighbour cells
        index2 = 0
        for ix in range(-ncells_c[0] + 1, ncells_c[0]):
            for iy in range(-ncells_c[1] + 1, ncells_c[1]):
                for iz in range(-ncells_c[2] + 1, ncells_c[2]):
                    i_c = np.array([ix, iy, iz])
                    pos_aav[index1][index2, :] = pos + np.dot(i_c, cell_cv)
                    index2 += 1
    return pos_aav


class Cavity(NeedsGD):
    """Base class for representing a cavity in the solvent.

    Attributes:
    g_g           -- The cavity on the fine grid. It varies from zero at the
                     solute location to one in the bulk solvent.
    del_g_del_n_g -- The partial derivative of the cavity with respect to
                     the electron density on the fine grid.
    grad_g_vg     -- The gradient of the cavity on the fine grid.
    V             -- global Volume in Bohr ** 3 or None
    A             -- global Area in Bohr ** 2 or None
    """

    def __init__(self, surface_calculator=None, volume_calculator=None):
        """Constructor for the Cavity class.

        Arguments:
        surface_calculator -- A SurfaceCalculator instance or None
        volume_calculator  -- A VolumeCalculator instance or None
        """
        NeedsGD.__init__(self)
        self.g_g = None
        self.del_g_del_n_g = None
        self.grad_g_vg = None
        self.surface_calculator = surface_calculator
        self.volume_calculator = volume_calculator
        self.V = None  # global Volume
        self.A = None  # global Surface

    def estimate_memory(self, mem):
        ngrids = 1 + self.depends_on_el_density
        mem.subnode('Distribution Function', ngrids * self.gd.bytecount())
        mem.subnode(
            'Gradient of Distribution Function', 3 * self.gd.bytecount()
        )
        if self.surface_calculator is not None:
            self.surface_calculator.estimate_memory(
                mem.subnode('Surface Calculator')
            )
        if self.volume_calculator is not None:
            self.volume_calculator.estimate_memory(
                mem.subnode('Volume Calculator')
            )

    def update(self, atoms, density):
        """Update the cavity and its gradient.

        atoms are None, iff they have not changed.

        Return whether the cavity has changed.
        """
        raise NotImplementedError()

    def update_vol_surf(self):
        """Update volume and surface."""
        if self.surface_calculator is not None:
            self.surface_calculator.update(self)
        if self.volume_calculator is not None:
            self.volume_calculator.update(self)

    def communicate_vol_surf(self, world):
        """Communicate global volume and surface."""
        if self.surface_calculator is not None:
            A = np.array([self.surface_calculator.A])
            self.gd.comm.sum(A)
            world.broadcast(A, 0)
            self.A = A[0]
        else:
            self.A = None
        if self.volume_calculator is not None:
            V = np.array([self.volume_calculator.V])
            self.gd.comm.sum(V)
            world.broadcast(V, 0)
            self.V = V[0]
        else:
            self.V = None

    def allocate(self):
        NeedsGD.allocate(self)
        self.g_g = self.gd.empty()
        self.grad_g_vg = self.gd.empty(3)
        if self.depends_on_el_density:
            self.del_g_del_n_g = self.gd.empty()
        if self.surface_calculator is not None:
            self.surface_calculator.allocate()
        if self.volume_calculator is not None:
            self.volume_calculator.allocate()

    def set_grid_descriptor(self, gd):
        NeedsGD.set_grid_descriptor(self, gd)
        if self.surface_calculator is not None:
            self.surface_calculator.set_grid_descriptor(gd)
        if self.volume_calculator is not None:
            self.volume_calculator.set_grid_descriptor(gd)

    def get_del_r_vg(self, atom_index, density):
        """Return spatial derivatives with respect to atomic position."""
        raise NotImplementedError()

    @property
    def depends_on_el_density(self):
        """Return whether the cavity depends on the electron density."""
        raise NotImplementedError()

    @property
    def depends_on_atomic_positions(self):
        """Return whether the cavity depends explicitly on atomic positions."""
        raise NotImplementedError()

    def __str__(self):
        s = 'Cavity: %s\n' % (self.__class__, )
        for calc, calcname in ((self.surface_calculator, 'Surface'),
                               (self.volume_calculator, 'Volume')):
            if calc is None:
                s += '  %s Calculator: None\n' % (calcname, )
            else:
                s += indent(str(calc))
        return s

    def update_atoms(self, atoms, log):
        """Inexpensive initialization when atoms change."""
        pass

    def summary(self, log):
        """Log cavity surface area and volume."""
        A = self.A * Bohr ** 2 if self.A is not None else 'not calculated (no calculator defined)'
        V = self.V * Bohr ** 3 if self.V is not None else 'not calculated (no calculator defined)'
        log('Cavity Surface Area: %s' % (A, ))
        log('Cavity Volume: %s' % (V, ))


class EffectivePotentialCavity(Cavity):
    """Cavity built from effective potential and Boltzmann distribution.

    See also
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).
    """

    def __init__(
        self,
        effective_potential,
        temperature,
        surface_calculator=None, volume_calculator=None
    ):
        """Constructor for the EffectivePotentialCavity class.

        Additional arguments not present in base Cavity class:
        effective_potential -- A Potential instance.
        temperature         -- Temperature for the Boltzmann distribution
                               in Kelvin.
        """
        Cavity.__init__(self, surface_calculator, volume_calculator)
        self.effective_potential = effective_potential
        self.temperature = float(temperature)
        self.minus_beta = -1. / (kB * temperature / Hartree)

    def estimate_memory(self, mem):
        Cavity.estimate_memory(self, mem)
        self.effective_potential.estimate_memory(
            mem.subnode('Effective Potential')
        )

    def set_grid_descriptor(self, gd):
        Cavity.set_grid_descriptor(self, gd)
        self.effective_potential.set_grid_descriptor(gd)

    def allocate(self):
        Cavity.allocate(self)
        self.effective_potential.allocate()

    def update(self, atoms, density):
        if not self.effective_potential.update(atoms, density):
            return False
        u_g = self.effective_potential.u_g
        np.exp(u_g * self.minus_beta, out=self.g_g)
        if self.depends_on_el_density:
            self.del_g_del_n_g.fill(self.minus_beta)
            self.del_g_del_n_g *= self.g_g
            self.del_g_del_n_g *= self.effective_potential.del_u_del_n_g
        # This class supports the GradientSurface API,
        # so we can use it for the gradient also
        grad_inner_vg = self.get_grad_inner()
        del_outer_del_inner = self.get_del_outer_del_inner()
        for i in (0, 1, 2):
            np.multiply(
                grad_inner_vg[i],
                del_outer_del_inner,
                self.grad_g_vg[i]
            )
        return True

    def get_del_r_vg(self, atom_index, density):
        u = self.effective_potential
        del_u_del_r_vg = u.get_del_r_vg(atom_index, density)
        # there should be no more NaNs now, but let's keep the hint
        ## asserts lim_(||r - r_atom|| -> 0) dg/du * du/dr_atom = 0
        #del_u_del_r_vg[np.isnan(del_u_del_r_vg)] = .0
        return self.minus_beta * self.g_g * del_u_del_r_vg

    @property
    def depends_on_el_density(self):
        return self.effective_potential.depends_on_el_density

    @property
    def depends_on_atomic_positions(self):
        return self.effective_potential.depends_on_atomic_positions

    def __str__(self):
        s = Cavity.__str__(self)
        s += indent(str(self.effective_potential))
        s += '  temperature: %s\n' % (self.temperature, )
        return s

    def update_atoms(self, atoms, log):
        self.effective_potential.update_atoms(atoms, log)

    # --- BEGIN GradientSurface API ---

    def get_grad_inner(self):
        return self.effective_potential.grad_u_vg

    def get_del_outer_del_inner(self):
        return self.minus_beta * self.g_g

    # --- END GradientSurface API ---


class Potential(NeedsGD):
    """Base class for describing an effective potential.

    Attributes:
    u_g           -- The potential on the fine grid in Hartree.
    grad_u_vg     -- The gradient of the potential of the fine grid.
    del_u_del_n_g -- Partial derivative with respect to the electron density.
    """

    def __init__(self):
        NeedsGD.__init__(self)
        self.u_g = None
        self.del_u_del_n_g = None
        self.grad_u_vg = None

    @property
    def depends_on_el_density(self):
        """Return whether the cavity depends on the electron density."""
        raise NotImplementedError()

    @property
    def depends_on_atomic_positions(self):
        """returns whether the cavity depends explicitly on atomic positions"""
        raise NotImplementedError()

    def estimate_memory(self, mem):
        ngrids = 1 + self.depends_on_el_density
        mem.subnode('Potential', ngrids * self.gd.bytecount())
        mem.subnode('Gradient of Potential', 3 * self.gd.bytecount())

    def allocate(self):
        NeedsGD.allocate(self)
        self.u_g = self.gd.empty()
        if self.depends_on_el_density:
            self.del_u_del_n_g = self.gd.empty()
        self.grad_u_vg = self.gd.empty(3)

    def update(self, atoms, density):
        """Update the potential and its gradient.

        atoms are None, iff they have not changed.

        Return whether the potential has changed.
        """
        raise NotImplementedError()

    def get_del_r_vg(self, atom_index, density):
        """Return spatial derivatives with respect to atomic position."""
        raise NotImplementedError()

    def __str__(self):
        return 'Potential: %s\n' % (self.__class__, )

    def update_atoms(self, atoms, log):
        """Inexpensive initialization when atoms change."""
        pass


class Power12Potential(Potential):
    """Inverse power law potential.

    An 1 / r ** 12 repulsive potential
    taking the value u0 at the atomic radius.

    See also
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).
    """
    depends_on_el_density = False
    depends_on_atomic_positions = True

    def __init__(self, atomic_radii, u0, pbc_cutoff=1e-6, tiny=1e-10):
        """Constructor for the Power12Potential class.

        Arguments:
        atomic_radii -- Callable mapping an ase.Atoms object
                        to an iterable of atomic radii in Angstroms.
        u0           -- Strength of the potential at the atomic radius in eV.
        pbc_cutoff   -- Cutoff in eV for including neighbor cells in
                        a calculation with periodic boundary conditions.
        """
        Potential.__init__(self)
        self.atomic_radii = atomic_radii
        self.u0 = float(u0)
        self.pbc_cutoff = float(pbc_cutoff)
        self.tiny = float(tiny)
        self.r12_a = None
        self.r_vg = None
        self.pos_aav = None
        self.del_u_del_r_vg = None
        self.atomic_radii_output = None
        self.symbols = None

    def estimate_memory(self, mem):
        Potential.estimate_memory(self, mem)
        nbytes = self.gd.bytecount()
        mem.subnode('Coordinates', 3 * nbytes)
        mem.subnode('Atomic Position Derivative', 3 * nbytes)

    def allocate(self):
        Potential.allocate(self)
        self.r_vg = self.gd.get_grid_point_coordinates()
        self.del_u_del_r_vg = self.gd.empty(3)

    def update(self, atoms, density):
        if atoms is None:
            return False
        self.r12_a = (self.atomic_radii_output / Bohr) ** 12
        r_cutoff = (self.r12_a.max() * self.u0 / self.pbc_cutoff) ** (1. / 12.)
        self.pos_aav = get_pbc_positions(atoms, r_cutoff)
        self.u_g.fill(.0)
        self.grad_u_vg.fill(.0)
        na = np.newaxis
        for index, pos_av in self.pos_aav.items():
            r12 = self.r12_a[index]
            for pos_v in pos_av:
                origin_vg = pos_v[:, na, na, na]
                r_diff_vg = self.r_vg - origin_vg
                r_diff2_g = (r_diff_vg ** 2).sum(0)
                r_diff2_g[r_diff2_g < self.tiny] = self.tiny
                u_g = r12 / r_diff2_g ** 6
                self.u_g += u_g
                u_g /= r_diff2_g
                r_diff_vg *= u_g[na, ...]
                self.grad_u_vg += r_diff_vg
        self.u_g *= self.u0 / Hartree
        self.grad_u_vg *= -12. * self.u0 / Hartree
        # avoid overflow in norm calculation:
        self.grad_u_vg[self.grad_u_vg < -1e20] = -1e20
        self.grad_u_vg[self.grad_u_vg > 1e20] = 1e20
        return True

    def get_del_r_vg(self, atom_index, density):
        u0 = self.u0 / Hartree
        r12 = self.r12_a[atom_index]
        na = np.newaxis
        self.del_u_del_r_vg.fill(.0)
        for pos_v in self.pos_aav[atom_index]:
            origin_vg = pos_v[:, na, na, na]
            diff_vg = self.r_vg - origin_vg
            diff2_g = (diff_vg ** 2).sum(0)
            diff2_g[diff2_g < self.tiny] = self.tiny
            diff2_g **= 7
            diff_vg /= diff2_g[na, ...]
            self.del_u_del_r_vg += diff_vg
        self.del_u_del_r_vg *= (12. * u0 * r12)
        return self.del_u_del_r_vg

    def __str__(self):
        s = Potential.__str__(self)
        s += '  atomic_radii: %s\n' % (self.atomic_radii, )
        s += '  u0: %s\n' % (self.u0, )
        s += '  pbc_cutoff: %s\n' % (self.pbc_cutoff, )
        s += '  tiny: %s\n' % (self.tiny, )
        return s

    def update_atoms(self, atoms, log):
        set_log_and_check_radii(self, atoms, log)


class SmoothStepCavity(Cavity):
    """Base class for cavities based on a smooth step function and a density.

    Attributes:
    del_g_del_rho_g -- Partial derivative with respect to the density.
    """

    def __init__(
        self,
        density,
        surface_calculator=None, volume_calculator=None,
    ):
        """Constructor for the SmoothStepCavity class.

        Additional arguments not present in the base Cavity class:
        density -- A Density instance
        """
        Cavity.__init__(self, surface_calculator, volume_calculator)
        self.del_g_del_rho_g = None
        self.density = density

    @property
    def depends_on_el_density(self):
        return self.density.depends_on_el_density

    @property
    def depends_on_atomic_positions(self):
        return self.density.depends_on_atomic_positions

    def set_grid_descriptor(self, gd):
        Cavity.set_grid_descriptor(self, gd)
        self.density.set_grid_descriptor(gd)

    def estimate_memory(self, mem):
        Cavity.estimate_memory(self, mem)
        mem.subnode('Cavity Derivative', self.gd.bytecount())
        self.density.estimate_memory(mem.subnode('Density'))

    def allocate(self):
        Cavity.allocate(self)
        self.del_g_del_rho_g = self.gd.empty()
        self.density.allocate()

    def update(self, atoms, density):
        if not self.density.update(atoms, density):
            return False
        self.update_smooth_step(self.density.rho_g)
        if self.depends_on_el_density:
            np.multiply(
                self.del_g_del_rho_g,
                self.density.del_rho_del_n_g,
                self.del_g_del_n_g
            )
        # This class supports the GradientSurface API,
        # so we can use it for the gradient also
        grad_inner_vg = self.get_grad_inner()
        del_outer_del_inner = self.get_del_outer_del_inner()
        for i in (0, 1, 2):
            np.multiply(
                grad_inner_vg[i],
                del_outer_del_inner,
                self.grad_g_vg[i]
                )
        return True

    def update_smooth_step(self, rho_g):
        """Calculate self.g_g and self.del_g_del_rho_g."""
        raise NotImplementedError()

    def get_del_r_vg(self, atom_index, density):
        return self.del_g_del_rho_g * self.density.get_del_r_vg(
            atom_index, density
        )

    def __str__(self):
        s = Cavity.__str__(self)
        s += indent(str(self.density))
        return s

    def update_atoms(self, atoms, log):
        self.density.update_atoms(atoms, log)

    # --- BEGIN GradientSurface API ---
    def get_grad_inner(self):
        return self.density.grad_rho_vg

    def get_del_outer_del_inner(self):
        return self.del_g_del_rho_g

    # --- END GradientSurface API ---


class Density(NeedsGD):
    """Class representing a density for the use with a SmoothStepCavity.

    Attributes:
    rho_g           -- The density on the fine grid in 1 / Bohr ** 3.
    del_rho_del_n_g -- Partial derivative with respect to the electron density.
    grad_rho_vg     -- The gradient of the density on the fine grid.
    """

    def __init__(self):
        NeedsGD.__init__(self)
        self.rho_g = None
        self.del_rho_del_n_g = None
        self.grad_rho_vg = None

    def estimate_memory(self, mem):
        nbytes = self.gd.bytecount()
        mem.subnode('Density', nbytes)
        if self.depends_on_el_density:
            mem.subnode('Density Derivative', nbytes)
        mem.subnode('Gradient of Density', 3 * nbytes)

    def allocate(self):
        NeedsGD.allocate(self)
        self.rho_g = self.gd.empty()
        if self.depends_on_el_density:
            self.del_rho_del_n_g = self.gd.empty()
        self.grad_rho_vg = self.gd.empty(3)

    def update(self, atoms, density):
        raise NotImplementedError()

    @property
    def depends_on_el_density(self):
        raise NotImplementedError()

    @property
    def depends_on_atomic_positions(self):
        raise NotImplementedError()

    def __str__(self):
        return "Density: %s\n" % (self.__class__, )

    def update_atoms(self, atoms, log):
        """Inexpensive initialization when atoms change."""
        pass


class FDGradientDensity(Density):
    """Base class for all Density classes with finite difference gradient"""

    def __init__(self, boundary_value, nn=3):
        """Constructur for the FDGradientDensity class.

        Arguments:
        boundary_value -- Boundary value of rho_g
                          (for non-periodic directions).
        nn             -- Stencil size for the finite difference gradient.
        """
        Density.__init__(self)
        self.boundary_value = boundary_value
        self.nn = nn
        self.gradient = None

    def allocate(self):
        Density.allocate(self)
        self.gradient = [
            Gradient(self.gd, i, 1.0, self.nn) for i in (0, 1, 2)
        ]

    def update(self, atoms, density):
        changed = self.update_only_density(atoms, density)
        if changed:
            self.update_gradient()
        return changed

    def update_gradient(self):
        if self.boundary_value != 0:
            in_g = self.rho_g - self.boundary_value
        else:
            in_g = self.rho_g
        for i in (0, 1, 2):
            self.gradient[i].apply(in_g, self.grad_rho_vg[i])


class ElDensity(FDGradientDensity):
    """Wrapper class for using the electron density in a SmoothStepCavity.

    (Hopefully small) negative values of the electron density are set to zero.
    """

    depends_on_el_density = True
    depends_on_atomic_positions = False

    def __init__(self, nn=3):
        """Constructor for the ElDensity class.

        Arguments:
        nn -- Stencil size for the finite difference gradient.
        """
        FDGradientDensity.__init__(self, boundary_value=.0, nn=nn)

    def allocate(self):
        FDGradientDensity.allocate(self)
        self.del_rho_del_n_g = 1.  # free array

    def update_only_density(self, atoms, density):
        self.rho_g[:] = density.nt_g
        self.rho_g[self.rho_g < .0] = .0
        return True


# TODO: implement analytic gradient for SSS09Density
class SSS09Density(FDGradientDensity):
    """Fake density from atomic radii for the use in a SmoothStepCavity.

    Following V. M. Sanchez, M. Sued and D. A. Scherlis,
    J. Chem. Phys. 131, 174108 (2009).
    """

    depends_on_el_density = False
    depends_on_atomic_positions = True

    def __init__(self, atomic_radii, pbc_cutoff=1e-3, nn=3, tiny=1e-100):
        """Constructor for the SSS09Density class.

        Arguments:
        atomic_radii -- Callable mapping an ase.Atoms object
                        to an iterable of atomic radii in Angstroms.
        pbc_cutoff   -- Cutoff in eV for including neighbor cells in
                        a calculation with periodic boundary conditions.
        nn           -- Stencil size for the finite difference gradient.
        """
        FDGradientDensity.__init__(self, boundary_value=.0, nn=nn)
        self.atomic_radii = atomic_radii
        self.atomic_radii_output = None
        self.symbols = None
        self.pbc_cutoff = float(pbc_cutoff)
        self.tiny = float(tiny)
        self.pos_aav = None
        self.r_vg = None
        self.del_rho_del_r_vg = None

    def estimate_memory(self, mem):
        FDGradientDensity.estimate_memory(self, mem)
        nbytes = self.gd.bytecount()
        mem.subnode('Coordinates', 3 * nbytes)
        mem.subnode('Atomic Position Derivative', 3 * nbytes)

    def allocate(self):
        FDGradientDensity.allocate(self)
        self.r_vg = self.gd.get_grid_point_coordinates()
        self.del_rho_del_r_vg = self.gd.empty(3)

    def update_only_density(self, atoms, density):
        if atoms is None:
            return False
        r_a = self.atomic_radii_output / Bohr
        r_cutoff = r_a.max() - np.log(self.pbc_cutoff)
        self.pos_aav = get_pbc_positions(atoms, r_cutoff)
        self.rho_g.fill(.0)
        na = np.newaxis
        for index, pos_av in self.pos_aav.items():
            for pos_v in pos_av:
                origin_vg = pos_v[:, na, na, na]
                r_diff_vg = self.r_vg - origin_vg
                norm_r_diff_g = (r_diff_vg ** 2).sum(0) ** .5
                self.rho_g += np.exp(r_a[index] - norm_r_diff_g)
        return True

    def get_del_r_vg(self, atom_index, density):
        r_a = self.atomic_radii_output[atom_index] / Bohr
        na = np.newaxis
        self.del_rho_del_r_vg.fill(.0)
        for pos_v in self.pos_aav[atom_index]:
            origin_vg = pos_v[:, na, na, na]
            r_diff_vg = self.r_vg - origin_vg
            norm_r_diff_g = np.sum(r_diff_vg ** 2, axis=0) ** .5
            norm_r_diff_g[norm_r_diff_g < self.tiny] = self.tiny
            exponential = np.exp(r_a - norm_r_diff_g)
            exponential /= norm_r_diff_g
            r_diff_vg *= exponential[na, ...]
            self.del_rho_del_r_vg += r_diff_vg
        return self.del_rho_del_r_vg

    def __str__(self):
        s = FDGradientDensity.__str__(self)
        s += '  atomic_radii: %s\n' % (self.atomic_radii, )
        s += '  pbc_cutoff: %s\n' % (self.pbc_cutoff, )
        s += '  tiny: %s\n' % (self.tiny, )
        return s

    def update_atoms(self, atoms, log):
        set_log_and_check_radii(self, atoms, log)


class ADM12SmoothStepCavity(SmoothStepCavity):
    """Cavity from smooth step function.

    Following O. Andreussi, I. Dabo, and N. Marzari,
    J. Chem. Phys. 136, 064102 (2012).
    """

    def __init__(
        self,
        rhomin, rhomax, epsinf,
        density,
        surface_calculator=None, volume_calculator=None
    ):
        """Constructor for the ADM12SmoothStepCavity class.

        Additional arguments not present in the SmoothStepCavity class:
        rhomin -- Lower density isovalue in 1 / Angstrom ** 3.
        rhomax -- Upper density isovalue in 1 / Angstrom ** 3.
        epsinf -- Static dielectric constant of the solvent.
        """
        SmoothStepCavity.__init__(
            self, density, surface_calculator, volume_calculator
        )
        self.rhomin = float(rhomin)
        self.rhomax = float(rhomax)
        self.epsinf = float(epsinf)

    def update_smooth_step(self, rho_g):
        eps = self.epsinf
        inside = rho_g > self.rhomax * Bohr ** 3
        outside = rho_g < self.rhomin * Bohr ** 3
        transition = np.logical_not(
            np.logical_or(inside, outside)
        )
        self.g_g[inside] = .0
        self.g_g[outside] = 1.
        self.del_g_del_rho_g.fill(.0)
        t, dt = self._get_t_dt(np.log(rho_g[transition]))
        if eps == 1.0:
            # lim_{eps -> 1} (eps - eps ** t) / (eps - 1) = 1 - t
            self.g_g[transition] = t
            self.del_g_del_rho_g[transition] = dt / rho_g[transition]
        else:
            eps_to_t = eps ** t
            self.g_g[transition] = (eps_to_t - 1.) / (eps - 1.)
            self.del_g_del_rho_g[transition] = (
                eps_to_t * np.log(eps) * dt
            ) / (
                rho_g[transition] * (eps - 1.)
            )

    def _get_t_dt(self, x):
        lnmax = np.log(self.rhomax * Bohr ** 3)
        lnmin = np.log(self.rhomin * Bohr ** 3)
        twopi = 2. * np.pi
        arg = twopi * (lnmax - x) / (lnmax - lnmin)
        t = (arg - np.sin(arg)) / twopi
        dt = -2. * np.sin(arg / 2.) ** 2 / (lnmax - lnmin)
        return (t, dt)

    def __str__(self):
        s = SmoothStepCavity.__str__(self)
        s += '  rhomin: %s\n' % (self.rhomin, )
        s += '  rhomax: %s\n' % (self.rhomax, )
        s += '  epsinf: %s\n' % (self.epsinf, )
        return s


class FG02SmoothStepCavity(SmoothStepCavity):
    """Cavity from smooth step function.

    Following J. Fattebert and F. Gygi, J. Comput. Chem. 23, 662 (2002).
    """

    def __init__(
        self,
        rho0,
        beta,
        density,
        surface_calculator=None, volume_calculator=None
    ):
        """Constructor for the FG02SmoothStepCavity class.

        Additional arguments not present in the SmoothStepCavity class:
        rho0 -- Density isovalue in 1 / Angstrom ** 3.
        beta -- Parameter controlling the steepness of the transition.
        """
        SmoothStepCavity.__init__(
            self, density, surface_calculator, volume_calculator
        )
        self.rho0 = float(rho0)
        self.beta = float(beta)

    def update_smooth_step(self, rho_g):
        rho0 = self.rho0 / (1. / Bohr ** 3)
        rho_scaled_g = rho_g / rho0
        exponent = 2. * self.beta
        np.divide(1., rho_scaled_g ** exponent + 1., self.g_g)
        np.multiply(
            (-exponent / rho0) * rho_scaled_g ** (exponent - 1.),
            self.g_g ** 2,
            self.del_g_del_rho_g
        )

    def __str__(self):
        s = SmoothStepCavity.__str__(self)
        s += '  rho0: %s\n' % (self.rho0, )
        s += '  beta: %s\n' % (self.beta, )
        return s


class SurfaceCalculator(NeedsGD):
    """Base class for surface calculators.

    Attributes:
    A                 -- Local area in Bohr ** 2.
    delta_A_delta_g_g -- Functional derivative with respect to the cavity
                         on the fine grid.
    """

    def __init__(self):
        NeedsGD.__init__(self)
        self.A = None
        self.delta_A_delta_g_g = None

    def estimate_memory(self, mem):
        mem.subnode('Functional Derivative', self.gd.bytecount())

    def allocate(self):
        NeedsGD.allocate(self)
        self.delta_A_delta_g_g = self.gd.empty()

    def __str__(self):
        return 'Surface Calculator: %s\n' % (self.__class__, )

    def update(self, cavity):
        """Calculate A and delta_A_delta_g_g."""
        raise NotImplementedError()


class GradientSurface(SurfaceCalculator):
    """Class for getting the surface area from the gradient of the cavity.

    See also W. Im, D. Beglov and B. Roux,
    Comput. Phys. Commun. 111, 59 (1998)
    and
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).
    """

    def __init__(self, nn=3):
        SurfaceCalculator.__init__(self)
        self.nn = nn
        self.gradient = None
        self.gradient_out = None
        self.norm_grad_out = None
        self.div_tmp = None

    def estimate_memory(self, mem):
        SurfaceCalculator.estimate_memory(self, mem)
        nbytes = self.gd.bytecount()
        mem.subnode('Gradient', 4 * nbytes)
        mem.subnode('Divergence', nbytes)

    def allocate(self):
        SurfaceCalculator.allocate(self)
        self.gradient = [
            Gradient(self.gd, i, 1.0, self.nn) for i in (0, 1, 2)
        ]
        self.gradient_out = self.gd.empty(3)
        self.norm_grad_out = self.gd.empty()
        self.div_tmp = self.gd.empty()

    def update(self, cavity):
        del_outer_del_inner = cavity.get_del_outer_del_inner()
        sign = np.sign(del_outer_del_inner.max() + del_outer_del_inner.min())
        self.gradient_out[...] = cavity.get_grad_inner()
        self.norm_grad_out = (self.gradient_out ** 2).sum(0) ** .5
        self.A = sign * self.gd.integrate(
            del_outer_del_inner * self.norm_grad_out, global_integral=False
        )
        mask = self.norm_grad_out > 1e-12  # avoid division by zero or overflow
        imask = np.logical_not(mask)
        masked_norm_grad = self.norm_grad_out[mask]
        for i in (0, 1, 2):
            self.gradient_out[i][mask] /= masked_norm_grad
            # set limit for later calculations:
            self.gradient_out[i][imask] = .0
        self.calc_div(self.gradient_out, self.delta_A_delta_g_g)
        if sign == 1:
            self.delta_A_delta_g_g *= -1.

    def calc_div(self, vec, out):
        self.gradient[0].apply(vec[0], out)
        self.gradient[1].apply(vec[1], self.div_tmp)
        out += self.div_tmp
        self.gradient[2].apply(vec[2], self.div_tmp)
        out += self.div_tmp


class VolumeCalculator(NeedsGD):
    """Base class for volume calculators.

    Attributes:
    V                 -- Local volume in Bohr ** 3.
    delta_V_delta_g_g -- Functional derivative with respect to the cavity
                         on the fine grid.
    """

    def __init__(self):
        NeedsGD.__init__(self)
        self.V = None
        self.delta_V_delta_g_g = None

    def estimate_memory(self, mem):
        mem.subnode('Functional Derivative', self.gd.bytecount())

    def allocate(self):
        NeedsGD.allocate(self)
        self.delta_V_delta_g_g = self.gd.empty()

    def __str__(self):
        return "Volume Calculator: %s\n" % (self.__class__, )

    def update(self, cavity):
        """Calculate V and delta_V_delta_g_g"""
        raise NotImplementedError()


class KB51Volume(VolumeCalculator):
    """KB51 Volume Calculator.

    V = Integral(1 - g) + kappa_T * k_B * T

    Following J. G. Kirkwood and F. P. Buff, J. Chem. Phys. 19, 6, 774 (1951).
    """

    def __init__(self, compressibility=.0, temperature=.0):
        """Constructor for KB51Volume class.

        Arguments:
        compressibility -- Isothermal compressibility of the solvent
                           in Angstrom ** 3 / eV.
        temperature     -- Temperature in Kelvin.
        """
        VolumeCalculator.__init__(self)
        self.compressibility = float(compressibility)
        self.temperature = float(temperature)

    def __str__(self):
        s = VolumeCalculator.__str__(self)
        s += '  compressibility: %s\n' % (self.compressibility, )
        s += '  temperature:     %s\n' % (self.temperature, )
        return s

    def allocate(self):
        VolumeCalculator.allocate(self)
        self.delta_V_delta_g_g = -1.  # frees array

    def update(self, cavity):
        self.V = self.gd.integrate(1. - cavity.g_g, global_integral=False)
        V_compress = self.compressibility * kB * self.temperature / Bohr ** 3
        self.V += V_compress / self.gd.comm.size
