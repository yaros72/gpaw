# General imports
import os
import numbers
import numpy as np
# ASE and GPAW imports
from ase.units import Bohr, Hartree
from gpaw.jellium import Jellium, JelliumSlab
from gpaw.hamiltonian import RealSpaceHamiltonian
from gpaw.fd_operators import Gradient
from gpaw.dipole_correction import DipoleCorrection

# Implicit solvent imports
from gpaw.solvation.cavity import (Potential, Power12Potential,
                                   get_pbc_positions)
from gpaw.solvation.calculator import SolvationGPAW
from gpaw.solvation.hamiltonian import SolvationRealSpaceHamiltonian
from gpaw.solvation.poisson import WeightedFDPoissonSolver


class SJM(SolvationGPAW):
    """Subclass of the SolvationGPAW class which includes the
       Solvated Jellium method.

       The method allows the simulation of an electrochemical environment
       by calculating constant potential quantities on the basis of constant
       charge DFT runs. For this purpose, it allows the usage of non-neutral
       periodic slab systems. Cell neutrality is achieved by adding a
       background charge in the solvent region above the slab

       Further detail are given in http://dx.doi.org/10.1021/acs.jpcc.8b02465

       Parameters:
        ne: float
            Number of electrons added in the atomic system and (with opposite
            sign) in the background charge region. At the start it can be an
            initial guess for the needed number of electrons and will be
            changed to the current number in the course of the calculation
        potential: float
            The potential that should be reached or kept in the course of the
            calculation. If set to "None" (default) a constant charge charge
            calculation based on the value of `ne` is performed.
        dpot: float
            Tolerance for the deviation of the input `potential`. If the
            potential is outside the defined range `ne` will be changed in
            order to get inside again.
        doublelayer: dict
            Parameters regarding the shape of the counter charge region
            Implemented keys:

            'start': float or 'cavity_like'
                If a float is given it corresponds to the lower
                boundary coordinate (default: z), where the counter charge
                starts. If 'cavity_like' is given the counter charge will
                take the form of the cavity up to the 'upper_limit'.
            'thickness': float
                Thickness of the counter charge region in Angstrom.
                Can only be used if start is not 'cavity_like' and will
                be overwritten by 'upper_limit'.
            'upper_limit': float
                Upper boundary of the counter charge region in terms of
                coordinate in Anfstrom (default: z). The default is
                atoms.cell[2][2] - 5.
        verbose: bool or 'cube'
            True:
                Write final electrostatic potential, background charge and
                and cavity into ASCII files.
            'cube':
                In addition to 'True', also write the cavity on the
                3D-grid into a cube file.


    """
    implemented_properties = ['energy', 'forces', 'stress', 'dipole',
                              'magmom', 'magmoms', 'ne', 'electrode_potential']

    def __init__(self, ne=0, doublelayer=None, potential=None,
                 dpot=0.01, tiny=1e-8, verbose=False, **gpaw_kwargs):

        SolvationGPAW.__init__(self, **gpaw_kwargs)

        self.tiny = tiny
        if abs(ne) < self.tiny:
            self.ne = self.tiny
        else:
            self.ne = ne

        self.potential = potential
        self.dpot = dpot
        self.dl = doublelayer
        self.verbose = verbose
        self.previous_ne = 0
        self.previous_pot = None
        self.slope = None

        self.log('-----------\nGPAW SJM module in %s\n----------\n'
                 % (os.path.abspath(__file__)))

    def create_hamiltonian(self, realspace, mode, xc):
        if not realspace:
            raise NotImplementedError(
                'SJM does not support '
                'calculations in reciprocal space yet.')

        dens = self.density

        self.hamiltonian = SJM_RealSpaceHamiltonian(
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

    def set_atoms(self, atoms):
        self.atoms = atoms
        self.spos_ac = atoms.get_scaled_positions() % 1.0

    def set(self, **kwargs):
        """Change parameters for calculator.

        It differs from the standard `set` function in two ways:
        - SJM specific keywords can be set
        - It does not reinitialize and delete `self.wfs` if the
          background charge is changed.

        """

        SJM_changes = {}
        for key in kwargs:
            if key in ['background_charge', 'ne', 'potential', 'dpot',
                       'doublelayer']:
                SJM_changes[key] = None

        for key in SJM_changes:
            SJM_changes[key] = kwargs.pop(key)

        major_changes = False
        if kwargs:
            SolvationGPAW.set(self, **kwargs)
            major_changes = True

        # SJM custom `set` for the new keywords
        for key in SJM_changes:

            if key in ['potential', 'doublelayer']:
                self.results = {}
                if key == 'potential':
                    self.potential = SJM_changes[key]
                    if self.potential is not None:
                        if self.wfs is None:
                            self.log('Target electrode potential: %1.4f V'
                                     % self.potential)
                        else:
                            self.log('New Target electrode potential: %1.4f V'
                                     % self.potential)
                    else:
                        self.log('Potential equilibration has been '
                                 'turned off')

                if key == 'doublelayer':
                    self.dl = SJM_changes[key]
                    self.set(background_charge=self.define_jellium(self.atoms))

            if key in ['dpot']:
                self.log('Potential tolerance has been changed to %1.4f V'
                         % SJM_changes[key])
                potint = self.get_electrode_potential()
                if abs(potint - self.potential) > SJM_changes[key]:
                    self.results = {}
                    self.log('Recalculating...\n')
                else:
                    self.log('Potential already reached the criterion.\n')
                self.dpot = SJM_changes[key]

            if key in ['background_charge']:
                self.parameters[key] = SJM_changes[key]
                self.log('------------')
                if self.wfs is not None:
                    if major_changes:
                        self.density = None
                    else:
                        self.density.reset()

                        self.density.background_charge = \
                            SJM_changes['background_charge']
                        self.density.background_charge.set_grid_descriptor(
                            self.density.finegd)

                        self.spos_ac = self.atoms.get_scaled_positions() % 1.0
                        self.initialize_positions(self.atoms)
                        self.wfs.initialize(self.density, self.hamiltonian,
                                            self.spos_ac)
                        self.wfs.eigensolver.reset()
                        self.scf.reset()

                        self.log('\n------------')
                        self.log('Jellium properties changed!')

                    if abs(self.ne) < self.tiny:
                        self.ne = self.tiny
                    self.wfs.nvalence += self.ne - self.previous_ne
                self.log('Current number of Excess Electrons: %1.5f' % self.ne)
                self.log('Jellium size parameters:')
                self.log('  Lower boundary: %s' % self.dl['start'])
                self.log('  Upper boundary: %s' % self.dl['upper_limit'])
                self.log('------------\n')

            if key in ['ne']:
                self.results = {}
                if abs(SJM_changes['ne']) < self.tiny:
                    self.ne = self.tiny
                else:
                    self.ne = SJM_changes['ne']

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['cell'], ):
        """
        Perform a calculation with SJM

        This module includes the potential equilibration loop characteristic
        for SJM. It is essentially a wrapper around GPAW.calculate()

        """

        if self.potential:
            for dummy in range(10):
                self.set_jellium(atoms)

                SolvationGPAW.calculate(self, atoms, ['energy'],
                                        system_changes)

                if self.verbose:
                    self.write_cavity_and_bckcharge()

                pot_int = self.get_electrode_potential()

                if abs(pot_int - self.potential) < self.dpot:
                    self.previous_pot = pot_int
                    self.previous_ne = self.ne
                    break

                if self.previous_pot is not None:
                    if abs(self.previous_ne - self.ne) > 2*self.tiny:
                        self.slope = (pot_int-self.previous_pot) / \
                            (self.ne-self.previous_ne)

                if self.slope is not None:
                    d = (pot_int-self.potential) - (self.slope*self.ne)
                    self.previous_ne = self.ne
                    self.ne = - d / self.slope
                else:
                    self.previous_ne = self.ne
                    self.ne += (
                        pot_int - self.potential) / \
                        abs(pot_int - self.potential) * self.dpot

                self.previous_pot = pot_int
            else:
                raise Exception(
                    'Potential could not be reached after ten iterations. '
                    'Aborting!')

        else:
            self.set_jellium(atoms)
            SolvationGPAW.calculate(self, atoms, ['energy'],
                                    system_changes)
            if self.verbose:
                self.write_cavity_and_bckcharge()
            self.previous_ne = self.ne

        if properties != ['energy']:
            SolvationGPAW.calculate(self, atoms, properties, [])

    def write_cavity_and_bckcharge(self):
        self.write_parallel_func_in_z(self.hamiltonian.cavity.g_g,
                                      name='cavity_')
        if self.verbose == 'cube':
            self.write_parallel_func_on_grid(self.hamiltonian.cavity.g_g,
                                             atoms=self.atoms,
                                             name='cavity')

        self.write_parallel_func_in_z(self.density.background_charge.mask_g,
                                      name='background_charge_')

    def summary(self):
        omega_extra = Hartree * self.hamiltonian.e_el_extrapolated + \
            self.get_electrode_potential() * self.ne
        omega_free = Hartree * self.hamiltonian.e_el_free + \
            self.get_electrode_potential()*self.ne

        self.results['energy'] = omega_extra
        self.results['free_energy'] = omega_free
        self.results['ne'] = self.ne
        self.results['electrode_potential'] = self.get_electrode_potential()
        self.hamiltonian.summary(self.occupations.fermilevel, self.log)

        self.log('----------------------------------------------------------')
        self.log("Grand Potential Energy (E_tot + E_solv - mu*ne):")
        self.log('Extrpol:    %s' % omega_extra)
        self.log('Free:    %s' % omega_free)
        self.log('-----------------------------------------------------------')

        self.density.summary(self.atoms, self.occupations.magmom, self.log)
        self.occupations.summary(self.log)
        self.wfs.summary(self.log)
        self.log.fd.flush()
        if self.verbose:
            self.write_parallel_func_in_z(self.hamiltonian.vHt_g * Hartree -
                                          self.get_fermi_level(),
                                          'elstat_potential_')

    def define_jellium(self, atoms):
        """Module for the definition of the explicit and counter charge

        """
        if self.dl is None:
            self.dl = {}

        if 'start' in self.dl:
            if self.dl['start'] == 'cavity_like':
                pass
            elif isinstance(self.dl['start'], numbers.Real):
                pass
            else:
                raise RuntimeError("The starting z value of the counter charge"
                                   "has to be either a number (coordinate),"
                                   "cavity_like' or not given (default: "
                                   " max(position)+3)")
        else:
            self.dl['start'] = max(atoms.positions[:, 2]) + 3.

        if 'upper_limit' in self.dl:
            pass
        elif 'thickness' in self.dl:
            if self.dl['start'] == 'cavity_like':
                raise RuntimeError("With a cavity-like counter charge only"
                                   "the keyword upper_limit(not thickness)"
                                   "can be used.")
            else:
                self.dl['upper_limit'] = self.dl['start'] + \
                    self.dl['thickness']
        else:
            self.dl['upper_limit'] = self.atoms.cell[2][2]-5.

        if self.dl['start'] == 'cavity_like':

            # XXX This part can definitely be improved
            if self.hamiltonian is None:
                filename = self.log.fd
                self.log.fd = None
                self.initialize(atoms)
                self.set_positions(atoms)
                self.log.fd = filename
                g_g = self.hamiltonian.cavity.g_g.copy()
                self.wfs = None
                self.density = None
                self.hamiltonian = None
                self.initialized = False
                return CavityShapedJellium(self.ne, g_g=g_g,
                                           z2=self.dl['upper_limit'])

            else:
                filename = self.log.fd
                self.log.fd = None
                self.set_positions(atoms)
                self.log.fd = filename
                return CavityShapedJellium(self.ne,
                                           g_g=self.hamiltonian.cavity.g_g,
                                           z2=self.dl['upper_limit'])

        elif isinstance(self.dl['start'], numbers.Real):
            return JelliumSlab(self.ne, z1=self.dl['start'],
                               z2=self.dl['upper_limit'])

    def get_electrode_potential(self):
        ham = self.hamiltonian
        fermilevel = self.occupations.fermilevel
        try:
            correction = ham.poisson.correction
        except AttributeError:
            wf2 = -fermilevel
        else:
            wf2 = (-fermilevel - correction) * Hartree

        return wf2  # refpot-E_f

    def set_jellium(self, atoms):
        if abs(self.previous_ne - self.ne) > 2*self.tiny:
            if abs(self.ne) < self.tiny:
                self.ne = self.tiny
            self.set(background_charge=self.define_jellium(atoms))

    """Various tools for writing global functions"""

    def write_parallel_func_in_z(self, g, name='g_z.out'):
        gd = self.density.finegd
        from gpaw.mpi import world
        G = gd.collect(g, broadcast=False)
        if world.rank == 0:
            G_z = G.mean(0).mean(0)
            out = open(name + self.log.fd.name.split('.')[0] + '.out', 'w')
            for i, val in enumerate(G_z):
                out.writelines('%f  %1.8f\n' % ((i + 1) * gd.h_cv[2][2] * Bohr,
                               val))
            out.close()

    def write_parallel_func_on_grid(self, g, atoms=None, name='func.cube',
                                    outstyle='cube'):
        from ase.io import write
        gd = self.density.finegd
        G = gd.collect(g, broadcast=False)
        if outstyle == 'cube':
            write(name + '.cube', atoms, data=G)
        elif outstyle == 'pckl':
            import pickle
            out = open(name, 'wb')
            pickle.dump(G, out)
            out.close()


class SJMPower12Potential(Power12Potential):
    """Inverse power law potential.

    An 1 / r ** 12 repulsive potential
    taking the value u0 at the atomic radius.

    See also
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).
    """
    depends_on_el_density = False
    depends_on_atomic_positions = True

    def __init__(self, atomic_radii, u0, pbc_cutoff=1e-6, tiny=1e-10,
                 H2O_layer=False, unsolv_backside=True):
        """Constructor for the SJMPower12Potential class.
        In SJM one also has the option of removing the solvent from the
        electrode backside and adding ghost plane/atoms to remove the solvent
        from the electrode-water interface.

        Parameters
        ----------
        atomic_radii : float
            Callable mapping an ase.Atoms object to an iterable of atomic radii
            in Angstroms.
        u0 : float
            Strength of the potential at the atomic radius in eV.
        pbc_cutoff : float
            Cutoff in eV for including neighbor cells in a calculation with
            periodic boundary conditions.
        H2O_layer: bool,int or 'plane' (default: False)
            True: Exclude the implicit solvent from the interface region
                between electrode and water. Ghost atoms will be added below
                the water layer.
            int: Explicitly account for the given number of water molecules
                above electrode. This is handy if H2O is directly adsorbed
                and a water layer is present in the unit cell at the same time.
            'plane': Use a plane instead of ghost atoms for freeing the
                surface.
        unsolv_backside: bool
            Exclude implicit solvent from the region behind the electrode

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
        self.H2O_layer = H2O_layer
        self.unsolv_backside = unsolv_backside

    def update(self, atoms, density):
        if atoms is None:
            return False
        self.r12_a = (self.atomic_radii_output / Bohr) ** 12
        r_cutoff = (self.r12_a.max() * self.u0 / self.pbc_cutoff) ** (1. / 12.)
        self.pos_aav = get_pbc_positions(atoms, r_cutoff)
        self.u_g.fill(.0)
        self.grad_u_vg.fill(.0)
        na = np.newaxis

        if self.unsolv_backside:
            # Removing solvent from electrode backside
            for z in range(self.u_g.shape[2]):
                if (self.r_vg[2, 0, 0, z] - atoms.positions[:, 2].min() /
                        Bohr < 0):
                    self.u_g[:, :, z] = np.inf
                    self.grad_u_vg[:, :, :, z] = 0

        if self.H2O_layer:
            # Add ghost coordinates and indices to pos_aav dictionary if
            # a water layer is present

            all_oxygen_ind = [atom.index for atom in atoms
                              if atom.symbol == 'O']

            # Disregard oxygens that don't belong to the water layer
            allwater_oxygen_ind = []
            for ox in all_oxygen_ind:
                nH = 0

                for i, atm in enumerate(atoms):
                    for period_atm in self.pos_aav[i]:
                        dist = period_atm * Bohr - atoms[ox].position
                        if np.linalg.norm(dist) < 1.1 and atm.symbol == 'H':
                            nH += 1

                if nH >= 2:
                    allwater_oxygen_ind.append(ox)

            # If the number of waters in the water layer is given as an input
            # (H2O_layer=i) then only the uppermost i water molecules are
            # regarded for unsolvating the interface (this is relevant if
            # water is adsorbed on the surface)
            if not isinstance(self.H2O_layer, (bool, str)):
                if self.H2O_layer % 1 < self.tiny:
                    self.H2O_layer = int(self.H2O_layer)
                else:
                    raise AttributeError('Only an integer number of water'
                                         'molecules is possible in the water'
                                         'layer')

                allwaters = atoms[allwater_oxygen_ind]
                indizes_water_ox_ind = np.argsort(allwaters.positions[:, 2],
                                                  axis=0)

                water_oxygen_ind = []
                for i in range(self.H2O_layer):
                    water_oxygen_ind.append(
                        allwater_oxygen_ind[indizes_water_ox_ind[-1-i]])

            else:
                water_oxygen_ind = allwater_oxygen_ind

            oxygen = self.pos_aav[water_oxygen_ind[0]] * Bohr
            if len(water_oxygen_ind) > 1:
                for windex in water_oxygen_ind[1:]:
                    oxygen = np.concatenate(
                        (oxygen, self.pos_aav[windex] * Bohr))

            O_layer = []
            if self.H2O_layer == 'plane':
                # Add a virtual plane
                # XXX:The value -1.5, being the amount of vdW radii of O in
                # distance of the plane relative to the oxygens in the water
                # layer, is an empirical one and should perhaps be
                # interchangable.
                # For some reason the poissonsolver has trouble converging
                # sometimes if this scheme is used

                plane_rel_oxygen = -1.5 * self.atomic_radii_output[
                    water_oxygen_ind[0]]
                plane_z = oxygen[:, 2].min() + plane_rel_oxygen

                r_diff_zg = self.r_vg[2, :, :, :] - plane_z / Bohr
                r_diff_zg[r_diff_zg < self.tiny] = self.tiny
                r_diff_zg = r_diff_zg ** 2
                u_g = self.r12_a[water_oxygen_ind[0]] / r_diff_zg ** 6
                self.u_g += u_g
                u_g /= r_diff_zg
                r_diff_zg *= u_g
                self.grad_u_vg[2, :, :, :] += r_diff_zg

            else:
                # Ghost atoms are added below the explicit water layer
                cell = atoms.cell.copy() / Bohr
                cell[2][2] = 1.
                natoms_in_plane = [round(np.linalg.norm(cell[0]) * 1.5),
                                   round(np.linalg.norm(cell[1]) * 1.5)]

                plane_z = (oxygen[:, 2].min() - 1.75 *
                           self.atomic_radii_output[water_oxygen_ind[0]])
                nghatoms_z = int(round(oxygen[:, 2].min() -
                                 atoms.positions[:, 2].min()))

                for i in range(int(natoms_in_plane[0])):
                    for j in range(int(natoms_in_plane[1])):
                        for k in np.linspace(atoms.positions[:, 2].min(),
                                             plane_z, num=nghatoms_z):

                            O_layer.append(np.dot(np.array(
                                [(1.5*i - natoms_in_plane[0]/4.) /
                                 natoms_in_plane[0],
                                 (1.5*j - natoms_in_plane[1]/4.) /
                                 natoms_in_plane[1],
                                 k / Bohr]), cell))

            # Add additional ghost O-atoms below the actual water O atoms
            # of water which frees the interface in case of corrugated
            # water layers
            for ox in oxygen / Bohr:
                O_layer.append([ox[0], ox[1], ox[2] - 1.0 *
                                self.atomic_radii_output[
                                    water_oxygen_ind[0]] / Bohr])

            r12_add = []
            for i in range(len(O_layer)):
                self.pos_aav[len(atoms) + i] = [O_layer[i]]
                r12_add.append(self.r12_a[water_oxygen_ind[0]])
            r12_add = np.array(r12_add)
            # r12_a must have same dimensions as pos_aav items
            self.r12_a = np.concatenate((self.r12_a, r12_add))

        for index, pos_av in self.pos_aav.items():
            pos_av = np.array(pos_av)
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
        self.grad_u_vg[self.grad_u_vg < -1e20] = -1e20
        self.grad_u_vg[self.grad_u_vg > 1e20] = 1e20

        return True


class SJM_RealSpaceHamiltonian(SolvationRealSpaceHamiltonian):
    """Realspace Hamiltonian with continuum solvent model in the context of SJM.

    See also Section III of
    A. Held and M. Walter, J. Chem. Phys. 141, 174108 (2014).

    In contrast to the standard implicit solvent model a dipole correction can
    also be applied.

    """

    def __init__(self, cavity, dielectric, interactions, gd, finegd, nspins,
                 setups, timer, xc, world, redistributor, vext=None,
                 psolver=None, stencil=3, collinear=None):
        """Constructor of SJM_RealSpaceHamiltonian class.


        Notes
        -----
        The only difference to SolvationRealSpaceHamiltonian is the
        possibility to perform a dipole correction

        """

        self.cavity = cavity
        self.dielectric = dielectric
        self.interactions = interactions
        cavity.set_grid_descriptor(finegd)
        dielectric.set_grid_descriptor(finegd)
        for ia in interactions:
            ia.set_grid_descriptor(finegd)

        if psolver is None:
            psolver = WeightedFDPoissonSolver()
            self.dipcorr = False
        elif isinstance(psolver, dict):
            psolver = SJMDipoleCorrection(WeightedFDPoissonSolver(),
                                          psolver['dipolelayer'])
            self.dipcorr = True

        if self.dipcorr:
            psolver.poissonsolver.set_dielectric(self.dielectric)
        else:
            psolver.set_dielectric(self.dielectric)

        self.gradient = None

        RealSpaceHamiltonian.__init__(
            self,
            gd, finegd, nspins, collinear, setups, timer, xc, world,
            vext=vext, psolver=psolver,
            stencil=stencil, redistributor=redistributor)

        for ia in interactions:
            setattr(self, 'e_' + ia.subscript, None)
        self.new_atoms = None
        self.vt_ia_g = None
        self.e_el_free = None
        self.e_el_extrapolated = None

    def initialize(self):
        if self.dipcorr:
            self.gradient = [Gradient(self.finegd, i, 1.0,
                             self.poisson.poissonsolver.nn)
                             for i in (0, 1, 2)]
        else:
            self.gradient = [Gradient(self.finegd, i, 1.0,
                             self.poisson.nn)
                             for i in (0, 1, 2)]

        self.vt_ia_g = self.finegd.zeros()
        self.cavity.allocate()
        self.dielectric.allocate()
        for ia in self.interactions:
            ia.allocate()
        RealSpaceHamiltonian.initialize(self)


"""Changed module which makes it possible to use the cavity shaped
   counter charge
"""


def create_background_charge(**kwargs):
    if 'z1' in kwargs:
        return JelliumSlab(**kwargs)
    elif 'g_g' in kwargs:
        return CavityShapedJellium(**kwargs)
    return Jellium(**kwargs)


class CavityShapedJellium(Jellium):
    """The Solvated Jellium object, where the counter charge takes the form
       of the cavity.
    """
    def __init__(self, charge, g_g, z2):
        """Put the positive background charge where the solvent is present and
           z < z2.

        Parameters:
        ----------

        g_g: array
            The g function from the implicit solvent model, representing the
            percentage of the actual dielectric constant on the grid.
        z2: float
            Position of upper surface in Angstrom units."""

        Jellium.__init__(self, charge)
        self.g_g = g_g
        self.z2 = (z2 - 0.0001) / Bohr

    def todict(self):
        dct = Jellium.todict(self)
        dct.update(z2=self.z2 * Bohr + 0.0001)
        return dct

    def get_mask(self):
        r_gv = self.gd.get_grid_point_coordinates().transpose((1, 2, 3, 0))
        mask = np.logical_not(r_gv[:, :, :, 2] > self.z2).astype(float)
        mask *= self.g_g
        return mask


class SJMDipoleCorrection(DipoleCorrection):
    """Dipole-correcting wrapper around another PoissonSolver specific for SJM.

    Iterative dipole correction class as applied in SJM.

    Notes
    -----

    The modules can easily be incorporated in the trunk version of GPAW
    by just adding the `fd_solv_solve`  and adapting the `solve` modules
    in the `DipoleCorrection` class.

    This module is currently calculating the correcting dipole potential
    iteratively and we would be very grateful if anybody could
    provide an analytical solution.

    New Parameters
    ---------
    corrterm: float
    Correction factor for the added countering dipole. This is calculated
    iteratively.

    last_corrterm: float
    Corrterm in the last iteration for getting the change of slope with change
    corrterm

    last_slope: float
    Same as for `last_corrterm`

    """
    def __init__(self, poissonsolver, direction, width=1.0):
        """Construct dipole correction object."""

        DipoleCorrection.__init__(self, poissonsolver, direction, width=1.0)
        self.corrterm = 1
        self.elcorr = None
        self.last_corrterm = None

    def solve(self, pot, dens, **kwargs):
        if isinstance(dens, np.ndarray):
            # finite-diference Poisson solver:
            if hasattr(self.poissonsolver, 'dielectric'):
                return self.fd_solv_solve(pot, dens, **kwargs)
            else:
                return self.fdsolve(pot, dens, **kwargs)
        # Plane-wave solver:
        self.pwsolve(pot, dens)

    def fd_solv_solve(self, vHt_g, rhot_g, **kwargs):

        gd = self.poissonsolver.gd
        slope_lim = 1e-8
        slope = slope_lim * 10

        dipmom = gd.calculate_dipole_moment(rhot_g)[2]

        if self.elcorr is not None:
            vHt_g[:, :] -= self.elcorr

        iters2 = self.poissonsolver.solve(vHt_g, rhot_g, **kwargs)

        sawtooth_z = self.sjm_sawtooth()
        L = gd.cell_cv[2, 2]

        while abs(slope) > slope_lim:
            vHt_g2 = vHt_g.copy()
            self.correction = 2 * np.pi * dipmom * L / \
                gd.volume * self.corrterm
            elcorr = -2 * self.correction

            elcorr *= sawtooth_z
            elcorr2 = elcorr[gd.beg_c[2]:gd.end_c[2]]
            vHt_g2[:, :] += elcorr2

            VHt_g = gd.collect(vHt_g2, broadcast=True)
            VHt_z = VHt_g.mean(0).mean(0)
            slope = VHt_z[2] - VHt_z[10]

            if abs(slope) > slope_lim:
                if self.last_corrterm is not None:
                    ds = (slope - self.last_slope) / \
                        (self.corrterm - self.last_corrterm)
                    con = slope - (ds * self.corrterm)
                    self.last_corrterm = self.corrterm
                    self.corrterm = -con / ds
                else:
                    self.last_corrterm = self.corrterm
                    self.corrterm -= slope * 10.
                self.last_slope = slope
            else:
                vHt_g[:, :] += elcorr2
                self.elcorr = elcorr2

        return iters2

    def sjm_sawtooth(self):
        gd = self.poissonsolver.gd
        c = self.c
        L = gd.cell_cv[c, c]
        step = gd.h_cv[c, c] / L

        eps_g = gd.collect(self.poissonsolver.dielectric.eps_gradeps[0],
                           broadcast=True)
        eps_z = eps_g.mean(0).mean(0)

        saw = np.zeros((int(L / gd.h_cv[c, c])))
        saw[0] = -0.5
        for i, eps in enumerate(eps_z):
            saw[i+1] = saw[i] + step / eps
        saw /= saw[-1] + step / eps_z[-1] - saw[0]
        saw -= (saw[0] + saw[-1] + step / eps_z[-1])/2.
        return saw
