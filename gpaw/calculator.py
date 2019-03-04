"""ASE-calculator interface."""
import warnings

import numpy as np
from ase.units import Bohr, Ha
from ase.calculators.calculator import Calculator, kpts2ndarray
from ase.utils import basestring, plural
from ase.utils.timing import Timer
from ase.dft.bandgap import bandgap

import gpaw
import gpaw.mpi as mpi
import gpaw.wavefunctions.pw as pw
from gpaw import dry_run, memory_estimate_depth
from gpaw.band_descriptor import BandDescriptor
from gpaw.density import RealSpaceDensity
from gpaw.eigensolvers import get_eigensolver
from gpaw.forces import calculate_forces
from gpaw.grid_descriptor import GridDescriptor
from gpaw.hamiltonian import RealSpaceHamiltonian
from gpaw.io.logger import GPAWLogger
from gpaw.io import Reader, Writer
from gpaw.jellium import create_background_charge
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.kpt_refine import create_kpoint_descriptor_with_refinement
from gpaw.kohnsham_layouts import get_KohnSham_layouts
from gpaw.matrix import suggest_blocking
from gpaw.occupations import create_occupation_number_object
from gpaw.output import (print_cell, print_positions,
                         print_parallelization_details)
from gpaw.paw import PAW
from gpaw.scf import SCFLoop
from gpaw.setup import Setups
from gpaw.symmetry import Symmetry
from gpaw.stress import calculate_stress
from gpaw.utilities import check_atoms_too_close
from gpaw.utilities.gpts import get_number_of_grid_points
from gpaw.utilities.grid import GridRedistributor
from gpaw.utilities.partition import AtomPartition
from gpaw.wavefunctions.mode import create_wave_function_mode
from gpaw.xc import XC
from gpaw.xc.sic import SIC


class GPAW(PAW, Calculator):
    """This is the ASE-calculator frontend for doing a PAW calculation."""

    implemented_properties = ['energy', 'forces', 'stress', 'dipole',
                              'magmom', 'magmoms']

    default_parameters = {
        'mode': 'fd',
        'xc': 'LDA',
        'occupations': None,
        'poissonsolver': None,
        'h': None,  # Angstrom
        'gpts': None,
        'kpts': [(0.0, 0.0, 0.0)],
        'nbands': None,
        'charge': 0,
        'setups': {},
        'basis': {},
        'spinpol': None,
        'fixdensity': False,
        'filter': None,
        'mixer': None,
        'eigensolver': None,
        'background_charge': None,
        'experimental': {'reuse_wfs_method': 'paw',
                         'niter_fixdensity': 0,
                         'magmoms': None,
                         'soc': None,
                         'kpt_refine': None},
        'external': None,
        'random': False,
        'hund': False,
        'maxiter': 333,
        'idiotproof': True,
        'symmetry': {'point_group': True,
                     'time_reversal': True,
                     'symmorphic': True,
                     'tolerance': 1e-7,
                     'do_not_symmetrize_the_density': False},
        'convergence': {'energy': 0.0005,  # eV / electron
                        'density': 1.0e-4,
                        'eigenstates': 4.0e-8,  # eV^2
                        'bands': 'occupied',
                        'forces': np.inf},  # eV / Ang
        'dtype': None,  # Deprecated
        'width': None,  # Deprecated
        'verbose': 0}

    default_parallel = {
        'kpt': None,
        'domain': gpaw.parsize_domain,
        'band': gpaw.parsize_bands,
        'order': 'kdb',
        'stridebands': False,
        'augment_grids': gpaw.augment_grids,
        'sl_auto': False,
        'sl_default': gpaw.sl_default,
        'sl_diagonalize': gpaw.sl_diagonalize,
        'sl_inverse_cholesky': gpaw.sl_inverse_cholesky,
        'sl_lcao': gpaw.sl_lcao,
        'sl_lrtddft': gpaw.sl_lrtddft,
        'use_elpa': False,
        'elpasolver': '2stage',
        'buffer_size': gpaw.buffer_size}

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None, timer=None,
                 communicator=None, txt='-', parallel=None, **kwargs):

        self.parallel = dict(self.default_parallel)
        if parallel:
            for key in parallel:
                if key not in self.default_parallel:
                    allowed = ', '.join(list(self.default_parallel.keys()))
                    raise TypeError('Unexpected keyword "{}" in "parallel" '
                                    'dictionary.  Must be one of: {}'
                                    .format(key, allowed))
            self.parallel.update(parallel)

        if timer is None:
            self.timer = Timer()
        else:
            self.timer = timer

        self.scf = None
        self.wfs = None
        self.occupations = None
        self.density = None
        self.hamiltonian = None
        self.spos_ac = None  # XXX store this in some better way.

        self.observers = []  # XXX move to self.scf
        self.initialized = False

        self.world = communicator
        if self.world is None:
            self.world = mpi.world
        elif not hasattr(self.world, 'new_communicator'):
            self.world = mpi.world.new_communicator(np.asarray(self.world))

        self.log = GPAWLogger(world=self.world)
        self.log.fd = txt

        self.reader = None

        Calculator.__init__(self, restart, ignore_bad_restart_file, label,
                            atoms, **kwargs)

    def __del__(self):
        # Write timings and close reader if necessary.

        # If we crashed in the constructor (e.g. a bad keyword), we may not
        # have the normally expected attributes:
        if hasattr(self, 'timer'):
            self.timer.write(self.log.fd)

        if hasattr(self, 'reader') and self.reader is not None:
            self.reader.close()

    def write(self, filename, mode=''):
        self.log('Writing to {} (mode={!r})\n'.format(filename, mode))
        writer = Writer(filename, self.world)
        self._write(writer, mode)
        writer.close()
        self.world.barrier()

    def _write(self, writer, mode):
        from ase.io.trajectory import write_atoms
        writer.write(version=1, gpaw_version=gpaw.__version__,
                     ha=Ha, bohr=Bohr)

        write_atoms(writer.child('atoms'), self.atoms)
        writer.child('results').write(**self.results)
        writer.child('parameters').write(**self.todict())

        self.density.write(writer.child('density'))
        self.hamiltonian.write(writer.child('hamiltonian'))
        self.occupations.write(writer.child('occupations'))
        self.scf.write(writer.child('scf'))
        self.wfs.write(writer.child('wave_functions'), mode == 'all')

        return writer

    def _set_atoms(self, atoms):
        check_atoms_too_close(atoms)
        self.atoms = atoms
        # GPAW works in terms of the scaled positions.  We want to
        # extract the scaled positions in only one place, and that is
        # here.  No other place may recalculate them, or we might end up
        # with rounding errors and inconsistencies.
        self.spos_ac = atoms.get_scaled_positions() % 1.0

    def read(self, filename):
        from ase.io.trajectory import read_atoms
        self.log('Reading from {}'.format(filename))

        self.reader = reader = Reader(filename)

        atoms = read_atoms(reader.atoms)
        self._set_atoms(atoms)

        res = reader.results
        self.results = dict((key, res.get(key)) for key in res.keys())
        if self.results:
            self.log('Read {}'.format(', '.join(sorted(self.results))))

        self.log('Reading input parameters:')
        # XXX param
        self.parameters = self.get_default_parameters()
        dct = {}
        for key, value in reader.parameters.asdict().items():
            if (isinstance(value, dict) and
                isinstance(self.parameters[key], dict)):
                self.parameters[key].update(value)
            else:
                self.parameters[key] = value
            dct[key] = self.parameters[key]

        self.log.print_dict(dct)
        self.log()

        self.initialize(reading=True)

        self.density.read(reader)
        self.hamiltonian.read(reader)
        self.occupations.read(reader)
        self.scf.read(reader)
        self.wfs.read(reader)

        # We need to do this in a better way:  XXX
        from gpaw.utilities.partition import AtomPartition
        atom_partition = AtomPartition(self.wfs.gd.comm,
                                       np.zeros(len(self.atoms), dtype=int))
        self.wfs.atom_partition = atom_partition
        self.density.atom_partition = atom_partition
        self.hamiltonian.atom_partition = atom_partition
        rank_a = self.density.gd.get_ranks_from_positions(self.spos_ac)
        new_atom_partition = AtomPartition(self.density.gd.comm, rank_a)
        for obj in [self.density, self.hamiltonian]:
            obj.set_positions_without_ruining_everything(self.spos_ac,
                                                         new_atom_partition)

        self.hamiltonian.xc.read(reader)

        if self.hamiltonian.xc.name == 'GLLBSC':
            # XXX GLLB: See test/lcaotddft/gllbsc.py
            self.occupations.calculate(self.wfs)

        return reader

    def check_state(self, atoms, tol=1e-15):
        system_changes = Calculator.check_state(self, atoms, tol)
        if 'positions' not in system_changes:
            if self.hamiltonian:
                if self.hamiltonian.vext:
                    if self.hamiltonian.vext.vext_g is None:
                        # QMMM atoms have moved:
                        system_changes.append('positions')
        return system_changes

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=['cell']):
        """Calculate things."""

        Calculator.calculate(self, atoms)
        atoms = self.atoms

        if system_changes:
            self.log('System changes:', ', '.join(system_changes), '\n')
            if system_changes == ['positions']:
                # Only positions have changed:
                self.density.reset()
            else:
                # Drastic changes:
                self.wfs = None
                self.occupations = None
                self.density = None
                self.hamiltonian = None
                self.scf = None
                self.initialize(atoms)

            self.set_positions(atoms)

        if not self.initialized:
            self.initialize(atoms)
            self.set_positions(atoms)

        if not (self.wfs.positions_set and self.hamiltonian.positions_set):
            self.set_positions(atoms)

        if not self.scf.converged:
            print_cell(self.wfs.gd, self.atoms.pbc, self.log)

            with self.timer('SCF-cycle'):
                self.scf.run(self.wfs, self.hamiltonian,
                             self.density, self.occupations,
                             self.log, self.call_observers)

            self.log('\nConverged after {} iterations.\n'
                     .format(self.scf.niter))

            e_free = self.hamiltonian.e_total_free
            e_extrapolated = self.hamiltonian.e_total_extrapolated
            self.results['energy'] = e_extrapolated * Ha
            self.results['free_energy'] = e_free * Ha

            dipole_v = self.density.calculate_dipole_moment() * Bohr
            self.log('Dipole moment: ({:.6f}, {:.6f}, {:.6f}) |e|*Ang\n'
                     .format(*dipole_v))
            self.results['dipole'] = dipole_v

            if self.wfs.nspins == 2 or not self.density.collinear:
                totmom_v, magmom_av = self.density.estimate_magnetic_moments()
                self.log('Total magnetic moment: ({:.6f}, {:.6f}, {:.6f})'
                         .format(*totmom_v))
                self.log('Local magnetic moments:')
                symbols = self.atoms.get_chemical_symbols()
                for a, mom_v in enumerate(magmom_av):
                    self.log('{:4} {:2} ({:9.6f}, {:9.6f}, {:9.6f})'
                             .format(a, symbols[a], *mom_v))
                self.log()
                self.results['magmom'] = self.occupations.magmom
                self.results['magmoms'] = magmom_av[:, 2].copy()

            self.summary()

            self.call_observers(self.scf.niter, final=True)

        if 'forces' in properties:
            with self.timer('Forces'):
                F_av = calculate_forces(self.wfs, self.density,
                                        self.hamiltonian, self.log)
                self.results['forces'] = F_av * (Ha / Bohr)

        if 'stress' in properties:
            with self.timer('Stress'):
                try:
                    stress = calculate_stress(self).flat[[0, 4, 8, 5, 2, 1]]
                except NotImplementedError:
                    # Our ASE Calculator base class will raise
                    # PropertyNotImplementedError for us.
                    pass
                else:
                    self.results['stress'] = stress * (Ha / Bohr**3)

    def summary(self):
        efermi = self.occupations.fermilevel
        self.hamiltonian.summary(efermi, self.log)
        self.density.summary(self.atoms, self.occupations.magmom, self.log)
        self.occupations.summary(self.log)
        self.wfs.summary(self.log)
        try:
            bandgap(self, output=self.log.fd, efermi=efermi * Ha)
        except ValueError:
            pass
        self.log.fd.flush()

    def set(self, **kwargs):
        """Change parameters for calculator.

        Examples::

            calc.set(xc='PBE')
            calc.set(nbands=20, kpts=(4, 1, 1))
        """

        # Verify that keys are consistent with default ones.
        for key in kwargs:
            if key != 'txt' and key not in self.default_parameters:
                raise TypeError('Unknown GPAW parameter: {}'.format(key))

            if key in ['convergence', 'symmetry',
                       'experimental'] and isinstance(kwargs[key], dict):
                # For values that are dictionaries, verify subkeys, too.
                default_dict = self.default_parameters[key]
                for subkey in kwargs[key]:
                    if subkey not in default_dict:
                        allowed = ', '.join(list(default_dict.keys()))
                        raise TypeError('Unknown subkeyword "{}" of keyword '
                                        '"{}".  Must be one of: {}'
                                        .format(subkey, key, allowed))

        changed_parameters = Calculator.set(self, **kwargs)

        for key in ['setups', 'basis']:
            if key in changed_parameters:
                dct = changed_parameters[key]
                if isinstance(dct, dict) and None in dct:
                    dct['default'] = dct.pop(None)
                    warnings.warn('Please use {key}={dct}'
                                  .format(key=key, dct=dct))

        # We need to handle txt early in order to get logging up and running:
        if 'txt' in changed_parameters:
            self.log.fd = changed_parameters.pop('txt')

        if not changed_parameters:
            return {}

        self.initialized = False
        self.scf = None
        self.results = {}

        self.log('Input parameters:')
        self.log.print_dict(changed_parameters)
        self.log()

        for key in changed_parameters:
            if key in ['eigensolver', 'convergence'] and self.wfs:
                self.wfs.set_eigensolver(None)

            if key in ['mixer', 'verbose', 'txt', 'hund', 'random',
                       'eigensolver', 'idiotproof']:
                continue

            if key in ['convergence', 'fixdensity', 'maxiter']:
                continue

            # Check nested arguments
            if key in ['experimental']:
                changed_parameters2 = changed_parameters[key]
                for key2 in changed_parameters2:
                    if key2 in ['kpt_refine', 'magmoms', 'soc']:
                        self.wfs = None
                    elif key2 in ['reuse_wfs_method', 'niter_fixdensity']:
                        continue
                    else:
                        raise TypeError('Unknown keyword argument:', key2)
                continue

            # More drastic changes:
            if self.wfs:
                self.wfs.set_orthonormalized(False)
            if key in ['external', 'xc', 'poissonsolver']:
                self.hamiltonian = None
            elif key in ['occupations', 'width']:
                pass
            elif key in ['charge', 'background_charge']:
                self.hamiltonian = None
                self.density = None
                self.wfs = None
            elif key in ['kpts', 'nbands', 'symmetry']:
                self.wfs = None
            elif key in ['h', 'gpts', 'setups', 'spinpol', 'dtype', 'mode']:
                self.density = None
                self.hamiltonian = None
                self.wfs = None
            elif key in ['basis']:
                self.wfs = None
            else:
                raise TypeError('Unknown keyword argument: "%s"' % key)

    def initialize_positions(self, atoms=None):
        """Update the positions of the atoms."""
        self.log('Initializing position-dependent things.\n')
        if atoms is None:
            atoms = self.atoms
        else:
            atoms = atoms.copy()
            self._set_atoms(atoms)

        mpi.synchronize_atoms(atoms, self.world)

        rank_a = self.wfs.gd.get_ranks_from_positions(self.spos_ac)
        atom_partition = AtomPartition(self.wfs.gd.comm, rank_a, name='gd')
        self.wfs.set_positions(self.spos_ac, atom_partition)
        self.density.set_positions(self.spos_ac, atom_partition)
        self.hamiltonian.set_positions(self.spos_ac, atom_partition)

    def set_positions(self, atoms=None):
        """Update the positions of the atoms and initialize wave functions."""
        self.initialize_positions(atoms)

        nlcao, nrand = self.wfs.initialize(self.density, self.hamiltonian,
                                           self.spos_ac)
        if nlcao + nrand:
            self.log('Creating initial wave functions:')
            if nlcao:
                self.log(' ', plural(nlcao, 'band'), 'from LCAO basis set')
            if nrand:
                self.log(' ', plural(nrand, 'band'), 'from random numbers')
            self.log()

        self.wfs.eigensolver.reset()
        self.scf.reset()
        print_positions(self.atoms, self.log, self.density.magmom_av)

    def initialize(self, atoms=None, reading=False):
        """Inexpensive initialization."""

        self.log('Initialize ...\n')

        if atoms is None:
            atoms = self.atoms
        else:
            atoms = atoms.copy()
            self._set_atoms(atoms)

        par = self.parameters

        natoms = len(atoms)

        cell_cv = atoms.get_cell() / Bohr
        number_of_lattice_vectors = cell_cv.any(axis=1).sum()
        if number_of_lattice_vectors < 3:
            raise ValueError(
                'GPAW requires 3 lattice vectors.  Your system has {}.'
                .format(number_of_lattice_vectors))

        pbc_c = atoms.get_pbc()
        assert len(pbc_c) == 3
        magmom_a = atoms.get_initial_magnetic_moments()

        if par.experimental.get('magmoms') is not None:
            magmom_av = np.array(par.experimental['magmoms'], float)
            collinear = False
        else:
            magmom_av = np.zeros((natoms, 3))
            magmom_av[:, 2] = magmom_a
            collinear = True

        mpi.synchronize_atoms(atoms, self.world)

        # Generate new xc functional only when it is reset by set
        # XXX sounds like this should use the _changed_keywords dictionary.
        if self.hamiltonian is None or self.hamiltonian.xc is None:
            if isinstance(par.xc, (basestring, dict)):
                xc = XC(par.xc, collinear=collinear, atoms=atoms)
            else:
                xc = par.xc
        else:
            xc = self.hamiltonian.xc

        mode = par.mode
        if isinstance(mode, basestring):
            mode = {'name': mode}
        if isinstance(mode, dict):
            mode = create_wave_function_mode(**mode)

        if par.dtype == complex:
            warnings.warn('Use mode={}(..., force_complex_dtype=True) '
                          'instead of dtype=complex'.format(mode.name.upper()))
            mode.force_complex_dtype = True
            del par['dtype']
            par.mode = mode

        if xc.orbital_dependent and mode.name == 'lcao':
            raise ValueError('LCAO mode does not support '
                             'orbital-dependent XC functionals.')

        realspace = (mode.name != 'pw' and mode.interpolation != 'fft')

        if not realspace:
            pbc_c = np.ones(3, bool)

        self.create_setups(mode, xc)

        if par.hund:
            if natoms != 1:
                raise ValueError('hund=True arg only valid for single atoms!')
            spinpol = True
            magmom_av[0, 2] = self.setups[0].get_hunds_rule_moment(par.charge)

        if collinear:
            magnetic = magmom_av.any()

            spinpol = par.spinpol
            if spinpol is None:
                spinpol = magnetic
            elif magnetic and not spinpol:
                raise ValueError('Non-zero initial magnetic moment for a ' +
                                 'spin-paired calculation!')
            nspins = 1 + int(spinpol)

            if spinpol:
                self.log('Spin-polarized calculation.')
                self.log('Magnetic moment: {:.6f}\n'.format(magmom_av.sum()))
            else:
                self.log('Spin-paired calculation\n')
        else:
            nspins = 1
            self.log('Non-collinear calculation.')
            self.log('Magnetic moment: ({:.6f}, {:.6f}, {:.6f})\n'
                     .format(*magmom_av.sum(0)))

        if isinstance(par.background_charge, dict):
            background = create_background_charge(**par.background_charge)
        else:
            background = par.background_charge

        nao = self.setups.nao
        nvalence = self.setups.nvalence - par.charge
        if par.background_charge is not None:
            nvalence += background.charge

        M = np.linalg.norm(magmom_av.sum(0))

        nbands = par.nbands

        orbital_free = any(setup.orbital_free for setup in self.setups)
        if orbital_free:
            nbands = 1

        if isinstance(nbands, basestring):
            if nbands == 'nao':
                nbands = nao
            elif nbands[-1] == '%':
                basebands = (nvalence + M) / 2
                nbands = int(np.ceil(float(nbands[:-1]) / 100 * basebands))
            else:
                raise ValueError('Integer expected: Only use a string '
                                 'if giving a percentage of occupied bands')

        if nbands is None:
            # Number of bound partial waves:
            nbandsmax = sum(setup.get_default_nbands()
                            for setup in self.setups)
            nbands = int(np.ceil((1.2 * (nvalence + M) / 2))) + 4
            if nbands > nbandsmax:
                nbands = nbandsmax
            if mode.name == 'lcao' and nbands > nao:
                nbands = nao
        elif nbands <= 0:
            nbands = max(1, int(nvalence + M + 0.5) // 2 + (-nbands))

        if nbands > nao and mode.name == 'lcao':
            raise ValueError('Too many bands for LCAO calculation: '
                             '%d bands and only %d atomic orbitals!' %
                             (nbands, nao))

        if nvalence < 0:
            raise ValueError(
                'Charge %f is not possible - not enough valence electrons' %
                par.charge)

        if nvalence > 2 * nbands and not orbital_free:
            raise ValueError('Too few bands!  Electrons: %f, bands: %d'
                             % (nvalence, nbands))

        self.create_occupations(magmom_av[:, 2].sum(), orbital_free)

        if self.scf is None:
            self.create_scf(nvalence, mode)

        self.create_symmetry(magmom_av, cell_cv)

        if not collinear:
            nbands *= 2

        if not self.wfs:
            self.create_wave_functions(mode, realspace,
                                       nspins, collinear, nbands, nao,
                                       nvalence, self.setups,
                                       cell_cv, pbc_c)
        else:
            self.wfs.set_setups(self.setups)

        if not self.wfs.eigensolver:
            self.create_eigensolver(xc, nbands, mode)

        if self.density is None and not reading:
            assert not par.fixdensity, 'No density to fix!'

        olddens = None
        if (self.density is not None and
            (self.density.gd.parsize_c != self.wfs.gd.parsize_c).any()):
            # Domain decomposition has changed, so we need to
            # reinitialize density and hamiltonian:
            if par.fixdensity:
                olddens = self.density

            self.density = None
            self.hamiltonian = None

        if self.density is None:
            self.create_density(realspace, mode, background)

        # XXXXXXXXXX if setups change, then setups.core_charge may change.
        # But that parameter was supplied in Density constructor!
        # This surely is a bug!
        self.density.initialize(self.setups, self.timer,
                                magmom_av, par.hund)
        self.density.set_mixer(par.mixer)
        if self.density.mixer.driver.name == 'dummy' or par.fixdensity:
            self.log('No density mixing\n')
        else:
            self.log(self.density.mixer, '\n')
        self.density.fixed = par.fixdensity
        self.density.log = self.log

        if olddens is not None:
            self.density.initialize_from_other_density(olddens,
                                                       self.wfs.kptband_comm)

        if self.hamiltonian is None:
            self.create_hamiltonian(realspace, mode, xc)

        xc.initialize(self.density, self.hamiltonian, self.wfs,
                      self.occupations)
        description = xc.get_description()
        if description is not None:
            self.log('XC parameters: {}\n'
                     .format('\n  '.join(description.splitlines())))

        if xc.name == 'GLLBSC' and olddens is not None:
            xc.heeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeelp(olddens)

        self.print_memory_estimate(maxdepth=memory_estimate_depth + 1)

        print_parallelization_details(self.wfs, self.hamiltonian, self.log)

        self.log('Number of atoms:', natoms)
        self.log('Number of atomic orbitals:', self.wfs.setups.nao)
        self.log('Number of bands in calculation:', self.wfs.bd.nbands)
        self.log('Bands to converge: ', end='')
        n = par.convergence.get('bands', 'occupied')
        if n == 'occupied':
            self.log('occupied states only')
        elif n == 'all':
            self.log('all')
        else:
            self.log('%d lowest bands' % n)
        self.log('Number of valence electrons:', self.wfs.nvalence)

        self.log(flush=True)

        self.timer.print_info(self)

        if dry_run:
            self.dry_run()

        if (realspace and
            self.hamiltonian.poisson.get_description() == 'FDTD+TDDFT'):
            self.hamiltonian.poisson.set_density(self.density)
            self.hamiltonian.poisson.print_messages(self.log)
            self.log.fd.flush()

        self.initialized = True
        self.log('... initialized\n')

    def create_setups(self, mode, xc):
        if self.parameters.filter is None and mode.name != 'pw':
            gamma = 1.6
            N_c = self.parameters.get('gpts')
            if N_c is None:
                h = (self.parameters.h or 0.2) / Bohr
            else:
                icell_vc = np.linalg.inv(self.atoms.cell)
                h = ((icell_vc**2).sum(0)**-0.5 / N_c).max() / Bohr

            def filter(rgd, rcut, f_r, l=0):
                gcut = np.pi / h - 2 / rcut / gamma
                ftmp = rgd.filter(f_r, rcut * gamma, gcut, l)
                f_r[:] = ftmp[:len(f_r)]
        else:
            filter = self.parameters.filter

        Z_a = self.atoms.get_atomic_numbers()
        self.setups = Setups(Z_a,
                             self.parameters.setups, self.parameters.basis,
                             xc, filter, self.world)
        self.log(self.setups)

    def create_grid_descriptor(self, N_c, cell_cv, pbc_c,
                               domain_comm, parsize_domain):
        return GridDescriptor(N_c, cell_cv, pbc_c, domain_comm,
                              parsize_domain)

    def create_occupations(self, magmom, orbital_free):
        occ = self.parameters.occupations

        if occ is None:
            if orbital_free:
                occ = {'name': 'orbital-free'}
            else:
                width = self.parameters.width
                if width is not None:
                    warnings.warn('Please use occupations=FermiDirac({})'
                                  .format(width))
                elif self.atoms.pbc.any():
                    width = 0.1  # eV
                else:
                    width = 0.0
                occ = {'name': 'fermi-dirac', 'width': width}

        if isinstance(occ, dict):
            occ = create_occupation_number_object(**occ)

        if self.parameters.fixdensity:
            occ.fixed_fermilevel = True
            if self.occupations:
                occ.fermilevel = self.occupations.fermilevel

        self.occupations = occ

        # If occupation numbers are changed, and we have wave functions,
        # recalculate the occupation numbers
        if self.wfs is not None:
            self.occupations.calculate(self.wfs)

        self.occupations.magmom = magmom

        self.log(self.occupations)

    def create_scf(self, nvalence, mode):
        # if mode.name == 'lcao':
        #     niter_fixdensity = 0
        # else:
        #     niter_fixdensity = 2

        nv = max(nvalence, 1)
        cc = self.parameters.convergence
        self.scf = SCFLoop(
            cc.get('eigenstates', 4.0e-8) / Ha**2 * nv,
            cc.get('energy', 0.0005) / Ha * nv,
            cc.get('density', 1.0e-4) * nv,
            cc.get('forces', np.inf) / (Ha / Bohr),
            self.parameters.maxiter,
            # XXX make sure niter_fixdensity value is *always* set from default
            # Subdictionary defaults seem to not be set when user provides
            # e.g. {}.  We should change that so it works like the ordinary
            # parameters.
            self.parameters.experimental.get('niter_fixdensity', 0),
            nv)
        self.log(self.scf)

    def create_symmetry(self, magmom_av, cell_cv):
        symm = self.parameters.symmetry
        if symm == 'off':
            symm = {'point_group': False, 'time_reversal': False}
        m_av = magmom_av.round(decimals=3)  # round off
        id_a = [id + tuple(m_v) for id, m_v in zip(self.setups.id_a, m_av)]
        self.symmetry = Symmetry(id_a, cell_cv, self.atoms.pbc, **symm)
        self.symmetry.analyze(self.spos_ac)
        self.setups.set_symmetry(self.symmetry)

    def create_eigensolver(self, xc, nbands, mode):
        # Number of bands to converge:
        nbands_converge = self.parameters.convergence.get('bands', 'occupied')
        if nbands_converge == 'all':
            nbands_converge = nbands
        elif nbands_converge != 'occupied':
            assert isinstance(nbands_converge, int)
            if nbands_converge < 0:
                nbands_converge += nbands
        eigensolver = get_eigensolver(self.parameters.eigensolver, mode,
                                      self.parameters.convergence)
        eigensolver.nbands_converge = nbands_converge
        # XXX Eigensolver class doesn't define an nbands_converge property

        if isinstance(xc, SIC):
            eigensolver.blocksize = 1

        self.wfs.set_eigensolver(eigensolver)

        self.log('Eigensolver\n  ', self.wfs.eigensolver, '\n')

    def create_density(self, realspace, mode, background):
        gd = self.wfs.gd

        big_gd = gd.new_descriptor(comm=self.world)
        # Check whether grid is too small.  8 is smallest admissible.
        # (we decide this by how difficult it is to make the tests pass)
        # (Actually it depends on stencils!  But let the user deal with it)
        N_c = big_gd.get_size_of_global_array(pad=True)
        too_small = np.any(N_c / big_gd.parsize_c < 8)
        if (self.parallel['augment_grids'] and not too_small and
            mode.name != 'pw'):
            aux_gd = big_gd
        else:
            aux_gd = gd

        redistributor = GridRedistributor(self.world,
                                          self.wfs.kptband_comm,
                                          gd, aux_gd)

        # Construct grid descriptor for fine grids for densities
        # and potentials:
        finegd = aux_gd.refine()

        kwargs = dict(
            gd=gd, finegd=finegd,
            nspins=self.wfs.nspins,
            collinear=self.wfs.collinear,
            charge=self.parameters.charge + self.wfs.setups.core_charge,
            redistributor=redistributor,
            background_charge=background)

        if realspace:
            self.density = RealSpaceDensity(stencil=mode.interpolation,
                                            **kwargs)
        else:
            self.density = pw.ReciprocalSpaceDensity(**kwargs)

        self.log(self.density, '\n')

    def create_hamiltonian(self, realspace, mode, xc):
        dens = self.density
        kwargs = dict(
            gd=dens.gd, finegd=dens.finegd,
            nspins=dens.nspins,
            collinear=dens.collinear,
            setups=dens.setups,
            timer=self.timer,
            xc=xc,
            world=self.world,
            redistributor=dens.redistributor,
            vext=self.parameters.external,
            psolver=self.parameters.poissonsolver)
        if realspace:
            self.hamiltonian = RealSpaceHamiltonian(stencil=mode.interpolation,
                                                    **kwargs)
            xc.set_grid_descriptor(self.hamiltonian.finegd)
        else:
            # This code will work if dens.redistributor uses
            # ordinary density.gd as aux_gd
            gd = dens.finegd

            xc_redist = None
            if self.parallel['augment_grids']:
                from gpaw.grid_descriptor import BadGridError
                try:
                    aux_gd = gd.new_descriptor(comm=self.world)
                except BadGridError as err:
                    import warnings
                    warnings.warn('Ignoring augment_grids: {}'
                                  .format(err))
                else:
                    bcast_comm = dens.redistributor.broadcast_comm
                    xc_redist = GridRedistributor(self.world, bcast_comm,
                                                  gd, aux_gd)

            self.hamiltonian = pw.ReciprocalSpaceHamiltonian(
                pd2=dens.pd2, pd3=dens.pd3, realpbc_c=self.atoms.pbc,
                xc_redistributor=xc_redist,
                **kwargs)
            xc.set_grid_descriptor(self.hamiltonian.xc_gd)

        self.hamiltonian.soc = self.parameters.experimental.get('soc')
        self.log(self.hamiltonian, '\n')

    def create_kpoint_descriptor(self, nspins):
        par = self.parameters

        bzkpts_kc = kpts2ndarray(par.kpts, self.atoms)
        kpt_refine = par.experimental.get('kpt_refine')
        if kpt_refine is None:
            kd = KPointDescriptor(bzkpts_kc, nspins)

            self.timer.start('Set symmetry')
            kd.set_symmetry(self.atoms, self.symmetry, comm=self.world)
            self.timer.stop('Set symmetry')

        else:
            self.timer.start('Set k-point refinement')
            kd = create_kpoint_descriptor_with_refinement(
                kpt_refine,
                bzkpts_kc, nspins, self.atoms,
                self.symmetry, comm=self.world,
                timer=self.timer)
            self.timer.stop('Set k-point refinement')
            # Update quantities which might have changed, if symmetry
            # was changed
            self.symmetry = kd.symmetry
            self.setups.set_symmetry(kd.symmetry)

        self.log(kd)

        return kd

    def create_wave_functions(self, mode, realspace,
                              nspins, collinear, nbands, nao, nvalence,
                              setups, cell_cv, pbc_c):
        par = self.parameters

        kd = self.create_kpoint_descriptor(nspins)

        parallelization = mpi.Parallelization(self.world,
                                              nspins * kd.nibzkpts)

        parsize_kpt = self.parallel['kpt']
        parsize_domain = self.parallel['domain']
        parsize_bands = self.parallel['band']

        ndomains = None
        if parsize_domain is not None:
            ndomains = np.prod(parsize_domain)
        parallelization.set(kpt=parsize_kpt,
                            domain=ndomains,
                            band=parsize_bands)
        comms = parallelization.build_communicators()
        domain_comm = comms['d']
        kpt_comm = comms['k']
        band_comm = comms['b']
        kptband_comm = comms['D']
        domainband_comm = comms['K']

        self.comms = comms

        if par.gpts is not None:
            if par.h is not None:
                raise ValueError("""You can't use both "gpts" and "h"!""")
            N_c = np.array(par.gpts)
        else:
            h = par.h
            if h is not None:
                h /= Bohr
            N_c = get_number_of_grid_points(cell_cv, h, mode, realspace,
                                            kd.symmetry)

        self.symmetry.check_grid(N_c)

        kd.set_communicator(kpt_comm)

        parstride_bands = self.parallel['stridebands']

        bd = BandDescriptor(nbands, band_comm, parstride_bands)

        # Construct grid descriptor for coarse grids for wave functions:
        gd = self.create_grid_descriptor(N_c, cell_cv, pbc_c,
                                         domain_comm, parsize_domain)

        if hasattr(self, 'time') or mode.force_complex_dtype or not collinear:
            dtype = complex
        else:
            if kd.gamma:
                dtype = float
            else:
                dtype = complex

        wfs_kwargs = dict(gd=gd, nvalence=nvalence, setups=setups,
                          bd=bd, dtype=dtype, world=self.world, kd=kd,
                          kptband_comm=kptband_comm, timer=self.timer)

        if self.parallel['sl_auto']:
            # Choose scalapack parallelization automatically

            for key, val in self.parallel.items():
                if (key.startswith('sl_') and key != 'sl_auto' and
                    val is not None):
                    raise ValueError("Cannot use 'sl_auto' together "
                                     "with '%s'" % key)

            max_scalapack_cpus = bd.comm.size * gd.comm.size
            sl_default = suggest_blocking(nbands, max_scalapack_cpus)
        else:
            sl_default = self.parallel['sl_default']

        if mode.name == 'lcao':
            assert collinear
            # Layouts used for general diagonalizer
            sl_lcao = self.parallel['sl_lcao']
            if sl_lcao is None:
                sl_lcao = sl_default

            elpasolver = None
            if self.parallel['use_elpa']:
                elpasolver = self.parallel['elpasolver']
            lcaoksl = get_KohnSham_layouts(sl_lcao, 'lcao',
                                           gd, bd, domainband_comm, dtype,
                                           nao=nao, timer=self.timer,
                                           elpasolver=elpasolver)

            self.wfs = mode(lcaoksl, **wfs_kwargs)

        elif mode.name == 'fd' or mode.name == 'pw':
            # Use (at most) all available LCAO for initialization
            lcaonbands = min(nbands, nao)

            try:
                lcaobd = BandDescriptor(lcaonbands, band_comm,
                                        parstride_bands)
            except RuntimeError:
                initksl = None
            else:
                # Layouts used for general diagonalizer
                # (LCAO initialization)
                sl_lcao = self.parallel['sl_lcao']
                if sl_lcao is None:
                    sl_lcao = sl_default
                initksl = get_KohnSham_layouts(sl_lcao, 'lcao',
                                               gd, lcaobd, domainband_comm,
                                               dtype, nao=nao,
                                               timer=self.timer)

            reuse_wfs_method = par.experimental.get('reuse_wfs_method', 'paw')
            sl = (domainband_comm,) + (self.parallel['sl_diagonalize'] or
                                       sl_default or
                                       (1, 1, None))
            self.wfs = mode(sl, initksl,
                            reuse_wfs_method=reuse_wfs_method,
                            collinear=collinear,
                            **wfs_kwargs)
        else:
            self.wfs = mode(self, collinear=collinear, **wfs_kwargs)

        self.log(self.wfs, '\n')

    def dry_run(self):
        # Can be overridden like in gpaw.atom.atompaw
        print_cell(self.wfs.gd, self.atoms.pbc, self.log)
        print_positions(self.atoms, self.log, self.density.magmom_av)
        self.log.fd.flush()

        # Write timing info now before the interpreter shuts down:
        self.__del__()

        # Disable timing output during shut-down:
        del self.timer

        raise SystemExit
