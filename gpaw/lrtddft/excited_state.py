"""Excited state as calculator object."""

from __future__ import print_function
import os
import errno

import numpy as np
from ase.units import Hartree
from ase.utils import convert_string_to_fd
from ase.utils.timing import Timer

import gpaw.mpi as mpi
from gpaw import GPAW, __version__, restart
from gpaw.density import RealSpaceDensity
from gpaw.io.logger import GPAWLogger
from gpaw.lrtddft import LrTDDFT
from gpaw.lrtddft.finite_differences import FiniteDifference
from gpaw.utilities.blas import axpy
from gpaw.wavefunctions.lcao import LCAOWaveFunctions


class ExcitedState(GPAW):

    def __init__(self, lrtddft=None, index=0, d=0.001, txt=None,
                 parallel=0, communicator=None, name=None, restart=None):
        """ExcitedState object.
        parallel: Can be used to parallelize the numerical force calculation
        over images.
        """

        self.timer = Timer()
        self.atoms = None
        if isinstance(index, int):
            self.index = UnconstraintIndex(index)
        else:
            self.index = index

        self.results = {}
        self.results['forces'] = None
        self.results['energy'] = None
        if communicator is None:
            try:
                communicator = lrtddft.calculator.wfs.world
            except:
                communicator = mpi.world
        self.world = communicator

        if restart is not None:
            self.read(restart)
            if txt is None:
                self.txt = self.lrtddft.txt
            else:
                self.txt = convert_string_to_fd(txt, self.world)

        if lrtddft is not None:
            self.lrtddft = lrtddft
            self.calculator = self.lrtddft.calculator
            self.atoms = self.calculator.atoms
            self.parameters = self.calculator.parameters
            if txt is None:
                self.txt = self.lrtddft.txt
            else:
                self.txt = convert_string_to_fd(txt, self.world)

        self.d = d
        self.parallel = parallel
        self.name = name

        self.log = GPAWLogger(self.world)
        self.log.fd = self.txt
        self.reader = None
        self.calculator.log.fd = self.txt
        self.log('#', self.__class__.__name__, __version__)
        self.log('#', self.index)
        if name:
            self.log('name=' + name)
        self.log('# Force displacement:', self.d)
        self.log

    def __del__(self):
        self.timer.write(self.log.fd)

    def set(self, **kwargs):
        self.calculator.set(**kwargs)

    def set_positions(self, atoms):
        """Update the positions of the atoms."""

        self.atoms = atoms.copy()
        self.results['forces'] = None
        self.results['energy'] = None

    def write(self, filename, mode=''):

        try:
            os.makedirs(filename)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        self.calculator.write(filename=filename + '/' + filename, mode=mode)
        self.lrtddft.write(filename=filename + '/' + filename + '.lr.dat.gz',
                           fh=None)

        f = open(filename + '/' + filename + '.exst', 'w')
        f.write('# ' + self.__class__.__name__ + __version__ + '\n')
        f.write('Displacement: {0}'.format(self.d) + '\n')
        f.write('Index: ' + self.index.__class__.__name__ + '\n')
        for k, v in self.index.__dict__.items():
            f.write('{0}, {1}'.format(k, v) + '\n')
        f.close()

        mpi.world.barrier()

    def read(self, filename):

        self.lrtddft = LrTDDFT(filename + '/' + filename + '.lr.dat.gz')
        self.atoms, self.calculator = restart(
            filename + '/' + filename, communicator=self.world)
        E0 = self.calculator.get_potential_energy()

        f = open(filename + '/' + filename + '.exst', 'r')
        f.readline()
        self.d = f.readline().replace('\n', '').split()[1]
        indextype = f.readline().replace('\n', '').split()[1]
        if indextype == 'UnconstraintIndex':
            iex = int(f.readline().replace('\n', '').split()[1])
            self.index = UnconstraintIndex(iex)
        else:
            direction = f.readline().replace('\n', '').split()[1]
            if direction in [str(0), str(1), str(2)]:
                direction = int(direction)
            else:
                direction = None

            val = f.readline().replace('\n', '').split()
            if indextype == 'MinimalOSIndex':

                self.index = MinimalOSIndex(float(val[1]), direction)
            else:
                emin = float(val[2])
                emax = float(val[3].replace(']', ''))
                self.index = MaximalOSIndex([emin, emax], direction)

        index = self.index.apply(self.lrtddft)
        self.results['energy'] = E0 + self.lrtddft[index].energy * Hartree
        self.lrtddft.set_calculator(self.calculator)

    def calculation_required(self, atoms, quantities):
        if len(quantities) == 0:
            return False

        if self.atoms is None:
            return True

        elif (len(atoms) != len(self.atoms) or
              (atoms.get_atomic_numbers() !=
               self.atoms.get_atomic_numbers()).any() or
              (atoms.get_initial_magnetic_moments() !=
               self.atoms.get_initial_magnetic_moments()).any() or
              (atoms.get_cell() != self.atoms.get_cell()).any() or
              (atoms.get_pbc() != self.atoms.get_pbc()).any()):
            return True
        elif (atoms.get_positions() !=
              self.atoms.get_positions()).any():
            return True

        for quantity in ['energy', 'forces']:
            if quantity in quantities:
                quantities.remove(quantity)
                if self.results[quantity] is None:
                    return True
        return len(quantities) > 0

    def check_state(self, atoms, tol=1e-15):
        system_changes = GPAW.check_state(self.calculator, atoms, tol)
        return system_changes

    def get_potential_energy(self, atoms=None, force_consistent=None):
        """Evaluate potential energy for the given excitation."""

        if atoms is None:
            atoms = self.atoms

        if self.calculation_required(atoms, ['energy']):
            self.results['energy'] = self.calculate(atoms)

        return self.results['energy']

    def calculate(self, atoms):
        """Evaluate your energy if needed."""
        self.set_positions(atoms)

        self.calculator.calculate(atoms)
        E0 = self.calculator.get_potential_energy()
        atoms.set_calculator(self)

        if hasattr(self, 'density'):
            del(self.density)
        self.lrtddft.forced_update()
        self.lrtddft.diagonalize()

        index = self.index.apply(self.lrtddft)

        energy = E0 + self.lrtddft[index].energy * Hartree

        self.log('--------------------------')
        self.log('Excited state')
        self.log(self.index)
        self.log('Energy:   {0}'.format(energy))
        self.log()

        return energy

    def get_forces(self, atoms=None, save=False):
        """Get finite-difference forces
        If save = True, restartfiles for every displacement are given
        """
        if atoms is None:
            atoms = self.atoms

        if self.calculation_required(atoms, ['forces']):
            atoms.set_calculator(self)

            # do the ground state calculation to set all
            # ranks to the same density to start with
            E0 = self.calculate(atoms)

            finite = FiniteDifference(
                atoms=atoms,
                propertyfunction=atoms.get_potential_energy,
                save=save,
                name="excited_state", ending='.gpw',
                d=self.d, parallel=self.parallel)
            F_av = finite.run()

            self.set_positions(atoms)
            self.results['energy'] = E0
            self.results['forces'] = F_av
            if self.txt:
                self.log('Excited state forces in eV/Ang:')
                symbols = self.atoms.get_chemical_symbols()
                for a, symbol in enumerate(symbols):
                    self.log(('%3d %-2s %10.5f %10.5f %10.5f' %
                              ((a, symbol) +
                               tuple(self.results['forces'][a]))))
        return self.results['forces']

    def forces_indexn(self, index):
        """ If restartfiles are created from the force calculation,
        this function allows the calculation of forces for every
        excited state index.
        """
        atoms = self.atoms

        def reforce(self, name):
            excalc = ExcitedState(index=index, restart=name)
            return excalc.get_potential_energy()

        fd = FiniteDifference(
            atoms=atoms, save=True,
            propertyfunction=self.atoms.get_potential_energy,
            name="excited_state", ending='.gpw',
            d=self.d, parallel=0)
        atoms.set_calculator(self)

        return fd.restart(reforce)

    def get_stress(self, atoms):
        """Return the stress for the current state of the Atoms."""
        raise NotImplementedError

    def initialize_density(self, method='dipole'):
        if hasattr(self, 'density') and self.density.method == method:
            return

        gsdensity = self.calculator.density
        lr = self.lrtddft
        self.density = ExcitedStateDensity(
            gsdensity.gd, gsdensity.finegd, lr.kss.npspins,
            gsdensity.collinear,
            gsdensity.charge,
            method=method, redistributor=gsdensity.redistributor)
        index = self.index.apply(self.lrtddft)
        self.density.initialize(self.lrtddft, index)
        self.density.update(self.calculator.wfs)

    def get_pseudo_density(self, **kwargs):
        """Return pseudo-density array."""
        method = kwargs.pop('method', 'dipole')
        self.initialize_density(method)
        return GPAW.get_pseudo_density(self, **kwargs)

    def get_all_electron_density(self, **kwargs):
        """Return all electron density array."""
        method = kwargs.pop('method', 'dipole')
        self.initialize_density(method)
        return GPAW.get_all_electron_density(self, **kwargs)


class UnconstraintIndex:

    def __init__(self, index):
        self.index = index

    def apply(self, *argv):
        return self.index

    def __str__(self):
        return (self.__class__.__name__ + '(' + str(self.index) + ')')


class MinimalOSIndex:

    """
    Constraint on minimal oscillator strength.

    Searches for the first excitation that has a larger
    oscillator strength than the given minimum.

    direction:
        None: averaged (default)
        0, 1, 2: x, y, z
    """

    def __init__(self, fmin=0.02, direction=None):
        self.fmin = fmin
        self.direction = direction

    def apply(self, lrtddft):
        i = 0
        fmax = 0.
        idir = 0
        if self.direction is not None:
            idir = 1 + self.direction
        while i < len(lrtddft):
            ex = lrtddft[i]
            f = ex.get_oscillator_strength()[idir]
            fmax = max(f, fmax)
            if f > self.fmin:
                return i
            i += 1
        error = 'The intensity constraint |f| > ' + str(self.fmin) + ' '
        error += 'can not be satisfied (max(f) = ' + str(fmax) + ').'
        raise RuntimeError(error)


class MaximalOSIndex:

    """
    Select maximal oscillator strength.

    Searches for the excitation with maximal oscillator strength
    in a given energy range.

    energy_range:
        None: take all (default)
        [Emin, Emax]: take only transition in this energy range
        Emax: the same as [0, Emax]
    direction:
        None: averaged (default)
        0, 1, 2: x, y, z
    """

    def __init__(self, energy_range=None, direction=None):
        if energy_range is None:
            energy_range = np.array([0.0, 1.e32])
        elif isinstance(energy_range, (int, float)):
            energy_range = np.array([0.0, energy_range]) / Hartree
        self.energy_range = energy_range

        self.direction = direction

    def apply(self, lrtddft):
        index = None
        fmax = 0.
        idir = 0
        if self.direction is not None:
            idir = 1 + self.direction
        emin, emax = self.energy_range
        for i, ex in enumerate(lrtddft):
            f = ex.get_oscillator_strength()[idir]
            e = ex.get_energy()
            if e >= emin and e < emax and f > fmax:
                fmax = f
                index = i
        if index is None:
            raise RuntimeError('No transition in the energy range ' +
                               '[%g,%g]' % self.energy_range)
        return index


class ExcitedStateDensity(RealSpaceDensity):

    """Approximate excited state density object."""

    def __init__(self, *args, **kwargs):
        self.method = kwargs.pop('method', 'dipole')
        RealSpaceDensity.__init__(self, *args, **kwargs)
        self.lrtddft = None
        self.index = None
        self.gsdensity = None
        self.nbands = None
        self.wocc_sn = None
        self.wunocc_sn = None

    def initialize(self, lrtddft, index):
        self.lrtddft = lrtddft
        self.index = index

        calc = lrtddft.calculator
        self.gsdensity = calc.density
        self.gd = self.gsdensity.gd
        self.nbands = calc.wfs.bd.nbands

        # obtain weights
        ex = lrtddft[index]
        wocc_sn = np.zeros((self.nspins, self.nbands))
        wunocc_sn = np.zeros((self.nspins, self.nbands))
        for f, k in zip(ex.f, ex.kss):
            # XXX why not k.fij * k.energy / energy ???
            if self.method == 'dipole':
                erat = k.energy / ex.energy
            elif self.method == 'orthogonal':
                erat = 1.
            else:
                raise NotImplementedError(
                    'method should be either "dipole" or "orthogonal"')
            wocc_sn[k.pspin, k.i] += erat * f ** 2
            wunocc_sn[k.pspin, k.j] += erat * f ** 2
        self.wocc_sn = wocc_sn
        self.wunocc_sn = wunocc_sn

        RealSpaceDensity.initialize(
            self, calc.wfs.setups, calc.timer, None, False)

        self.set_positions(calc.spos_ac, calc.wfs.atom_partition)

        D_asp = {}
        for a, D_sp in self.gsdensity.D_asp.items():
            repeats = self.nspins // self.gsdensity.nspins
            # XXX does this work always?
            D_asp[a] = (1. * D_sp).repeat(repeats, axis=0)
        self.update_atomic_density_matrices(D_asp)

    def update(self, wfs):
        self.timer.start('Density')
        self.timer.start('Pseudo density')
        self.calculate_pseudo_density(wfs)
        self.timer.stop('Pseudo density')
        self.timer.start('Atomic density matrices')
        f_un = []
        for kpt in wfs.kpt_u:
            f_n = kpt.f_n - self.wocc_sn[kpt.s] + self.wunocc_sn[kpt.s]
            if self.nspins > self.gsdensity.nspins:
                f_n = kpt.f_n - self.wocc_sn[1] + self.wunocc_sn[1]
            f_un.append(f_n)
        wfs.calculate_atomic_density_matrices_with_occupation(self.D_asp,
                                                              f_un)
        self.timer.stop('Atomic density matrices')
        self.timer.start('Multipole moments')
        comp_charge, _Q_aL = self.calculate_multipole_moments()
        self.timer.stop('Multipole moments')

        if isinstance(wfs, LCAOWaveFunctions):
            self.timer.start('Normalize')
            self.normalize(comp_charge)
            self.timer.stop('Normalize')

        self.timer.stop('Density')

    def calculate_pseudo_density(self, wfs):
        """Calculate nt_sG from scratch.

        nt_sG will be equal to nct_G plus the contribution from
        wfs.add_to_density().
        """
        nvspins = wfs.kd.nspins
        npspins = self.nspins
        self.nt_xG = self.gd.zeros(self.ncomponents)

        for s in range(npspins):
            for kpt in wfs.kpt_u:
                if s == kpt.s or npspins > nvspins:
                    f_n = kpt.f_n / (1. + int(npspins > nvspins))
                    for f, psit_G in zip((f_n - self.wocc_sn[s] +
                                          self.wunocc_sn[s]),
                                         kpt.psit_nG):
                        axpy(f, psit_G ** 2, self.nt_sG[s])
        self.nt_sG[:] += self.nct_G
