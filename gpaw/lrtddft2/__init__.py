"""Module for linear response TDDFT class with indexed K-matrix storage."""

import os
import sys
import datetime
import glob

import numpy as np

import ase.units
from ase.utils import devnull

from gpaw.xc import XC


# a KS determinant with a single occ-uncc excitation
# from gpaw.lrtddft2.ks_singles import KohnShamSingleExcitation

# a list of KS determinants with single occ-uncc excitations
from gpaw.lrtddft2.ks_singles import KohnShamSingles

# Matrix Kip,jq <ia|f_Hxc|jq>
from gpaw.lrtddft2.k_matrix import Kmatrix

# a set of linear combinations of KS single determinants
from gpaw.lrtddft2.lr_transitions import LrtddftTransitions

# a linear combination of KS single determinants
# for a CW laser with Lorentzian width (in energy)
from gpaw.lrtddft2.lr_response import LrResponse

# communicators
from gpaw.lrtddft2.lr_communicators import LrCommunicators


class LrTDDFT2:
    """Linear response TDDFT (Casida) class with indexed K-matrix storage."""
    def __init__(self,
                 basefilename,
                 gs_calc,
                 fxc=None,
                 min_occ=None,
                 max_occ=None,
                 min_unocc=None,
                 max_unocc=None,
                 max_energy_diff=1e9,
                 recalculate=None,
                 lr_communicators=None,
                 txt='-'):
        """Initialize linear response TDDFT without calculating anything.

        Note: Does NOT support spin polarized calculations yet.

        Protip: If K_matrix file is too large and you keep running out of memory when trying to calculate spectrum or response wavefunction,
        you can try
        "split -l 100000 xxx.K_matrix.ddddddofDDDDDD xxx.K_matrix.ddddddofDDDDDD."


        Input parameters:

        basefilename
          All files associated with this calculation are stored as
          *<basefilename>.<extension>*

        gs_calc
          Ground state calculator (if you are using eh_communicator,
          you need to take care that calc has suitable dd_communicator.)

        fxc
          Name of the exchange-correlation kernel (fxc) used in calculation.
          (optional)

        min_occ
          Index of the first occupied state to be included in the calculation.
          (optional)

        max_occ
          Index of the last occupied state (inclusive) to be included in the
          calculation. (optional)

        min_unocc
          Index of the first unoccupied state to be included in the
          calculation. (optional)

        max_unocc
          Index of the last unoccupied state (inclusive) to be included in the
          calculation. (optional)

        max_energy_diff
          Noninteracting Kohn-Sham excitations above this value are not
          included in the calculation. Units: eV (optional)

        recalculate
          | Force recalculation.
          | 'eigen'  : recalculate only eigensystem (useful for on-the-fly
          |            spectrum calculations and convergence checking)
          | 'matrix' : recalculate matrix without solving the eigensystem
          | 'all'    : recalculate everything
          | None     : do not recalculate anything if not needed (default)

        lr_communicators
          Communicators for parallelizing over electron-hole pairs (i.e.,
          rows of K-matrix) and domain. Note that ground state calculator
          must have a matching (domain decomposition) communicator, which
          can be assured by using lr_communicators
          to create both communicators.

        txt
          Filename for text output
        """

        # Save input params
        self.basefilename = basefilename
        self.fxc_name = fxc
        self.xc = XC(self.fxc_name)
        self.min_occ = min_occ
        self.max_occ = max_occ
        self.min_unocc = min_unocc
        self.max_unocc = max_unocc
        self.max_energy_diff = max_energy_diff / ase.units.Hartree
        self.recalculate = recalculate
        # Don't init calculator yet if it's not needed (to save memory)
        self.calc = gs_calc
        self.calc_ready = False

        # FIXME: SUPPORT ALSO SPIN POLARIZED
        self.kpt_ind = 0

        # Input paramers?
        self.deriv_scale = 1e-5   # fxc finite difference step
        # ignore transition if population difference is below this value:
        self.min_pop_diff = 1e-3

        # set up communicators
        self.lr_comms = lr_communicators

        if self.lr_comms is None:
            self.lr_comms = LrCommunicators()
        self.lr_comms.initialize(gs_calc)

        # Init text output
        if self.lr_comms.parent_comm.rank == 0 and txt is not None:
            if txt == '-':
                self.txt = sys.stdout
            elif isinstance(txt, str):
                self.txt = open(txt, 'w')
            else:
                self.txt = txt
        elif self.calc is not None:
            self.txt = self.calc.log.fd
        else:
            self.txt = devnull

        # Check and set unset params

        # If min/max_occ/unocc were not given, initialized them to include
        # everything: min_occ/unocc => 0, max_occ/unocc to nubmer of wfs,
        # energy diff to numerical infinity
        nbands = len(self.calc.wfs.kpt_u[self.kpt_ind].f_n)
        if self.min_occ is None:
            self.min_occ = 0
        if self.min_unocc is None:
            self.min_unocc = self.min_occ
        if self.max_occ is None:
            self.max_occ = nbands - 1
        if self.max_unocc is None:
            self.max_unocc = self.max_occ
        if self.max_energy_diff is None:
            self.max_energy_diff = 1e9

        self.min_occ = max(self.min_occ, 0)
        self.min_unocc = max(self.min_unocc, 0)

        if self.max_occ >= nbands:
            raise RuntimeError('Error in LrTDDFT2: max_occ >= nbands')
        if self.max_unocc >= nbands:
            raise RuntimeError('Error in LrTDDFT2: max_unocc >= nbands')

        # Only spin unpolarized calculations are supported atm
        # > FIXME
        assert len(self.calc.wfs.kpt_u) == 1, \
            'LrTDDFT2 does not support more than one k-point/spin.'
        self.kpt_ind = 0
        # <

        # Internal classes

        # list of singly excited Kohn-Sham Slater determinants
        # (ascending KS energy difference)
        self.ks_singles = KohnShamSingles(self.basefilename,
                                          self.calc,
                                          self.kpt_ind,
                                          self.min_occ, self.max_occ,
                                          self.min_unocc, self.max_unocc,
                                          self.max_energy_diff,
                                          self.min_pop_diff,
                                          self.lr_comms,
                                          self.txt)

        # Response kernel matrix K = <ip|f_Hxc|jq>
        # Note: this is not the Casida matrix
        self.K_matrix = Kmatrix(self.ks_singles, self.xc, self.deriv_scale)

        self.sl_lrtddft = self.calc.parallel['sl_lrtddft']

        # LR-TDDFT transitions
        self.lr_transitions = LrtddftTransitions(self.ks_singles,
                                                 self.K_matrix,
                                                 self.sl_lrtddft)

        # Response wavefunction
        self.lr_response_wf = None

        # Pair density
        self.pair_density = None  # pair density class

        # Timer
        self.timer = self.calc.timer
        self.timer.start('LrTDDFT')

        # If a previous calculation exist, read info file
        # self.read(self.basefilename)

        # Write info file
        # self.parent_comm.barrier()
        # if self.parent_comm.rank == 0:
        #     self.write_info(self.basefilename)

        # self.custom_axes = None
        # self.K_matrix_scaling_factor = 1.0
        self.K_matrix_values_ready = False

    def get_transitions(self, filename=None, min_energy=0.0, max_energy=30.0, units='eVcgs'):
        """Get transitions: energy, dipole strength and rotatory strength.

        Returns transitions as (w,S,R, Sx,Sy,Sz) where
        w is an array of frequencies,
        S is an array of corresponding dipole strengths,
        and R is an array of corresponding rotatory strengths.

        Input parameters:

        min_energy
          Minimum energy

        min_energy
          Maximum energy

        units
          Units for spectrum: 'au' or 'eVcgs'
        """

        self.calculate()
        self.txt.write('Calculating transitions (%s).\n' % str(datetime.datetime.now()))
        trans = self.lr_transitions.get_transitions(filename, min_energy, max_energy, units)
        self.txt.write('Transitions calculated  (%s).\n' % str(datetime.datetime.now()))
        return trans

    #################################################################
    def get_spectrum(self, filename=None, min_energy=0.0, max_energy=30.0,
                     energy_step=0.01, width=0.1, units='eVcgs'):
        """Get spectrum for dipole and rotatory strength.

        Returns folded spectrum as (w,S,R) where w is an array of frequencies,
        S is an array of corresponding dipole strengths, and R is an array of
        corresponding rotatory strengths.

        Input parameters:

        min_energy
          Minimum energy

        min_energy
          Maximum energy

        energy_step
          Spacing between calculated energies

        width
          Width of the Gaussian

        units
          Units for spectrum: 'au' or 'eVcgs'
        """
        self.calculate()
        self.txt.write('Calculating spectrum    (%s).\n' % str(datetime.datetime.now()))
        spec = self.lr_transitions.get_spectrum(filename, min_energy, max_energy, energy_step, width, units)
        self.txt.write('Spectrum calculated     (%s).\n' % str(datetime.datetime.now()))
        return spec

    #################################################################
    def get_transition_contributions(self, index_of_transition):
        """Get contributions of Kohn-Sham singles to a given transition
        as number of electrons contributing.

        Includes population difference.

        This method is meant to be used for small number of transitions.
        It is not suitable for analysing densely packed transitions of
        large systems. Use transition contribution map (TCM) or similar
        approach for this.

        Input parameters:

        index_of_transition:
          index of transition starting from zero
        """
        self.calculate()
        return self.lr_transitions.get_transition_contributions(index_of_transition)

    def calculate_response(self, excitation_energy, excitation_direction, lorentzian_width, units='eVang'):
        """Calculates and returns response using TD-DFPT.

        Input parameters:

        excitation_energy
          Energy of the laser in given units

        excitation_direction
          Vector for direction (will be normalized)

        lorentzian_width
          Life time or width parameter. Larger width results in wider
          energy envelope around excitation energy.
        """

        # S_z(omega) = 2 * omega sum_ip n_ip C^(im)_ip(omega) * mu^(z)_ip

        self.calculate()

        omega_au = excitation_energy
        width_au = lorentzian_width
        # always unit field in au !!!
        direction_au = np.array(excitation_direction)
        direction_au = direction_au / np.sqrt(np.vdot(direction_au,direction_au))

        if units == 'au':
            pass
        elif units == 'eVang':
            omega_au /= ase.units.Hartree
            width_au /= ase.units.Hartree
        else:
            raise RuntimeError('Error in calculate_response_wavefunction: Invalid units.')

        lr_response = LrResponse(self, omega_au, direction_au, width_au, self.sl_lrtddft)
        lr_response.solve()

        return lr_response

    def calculate(self):
        """Calculates linear response matrix and properties of KS
        electron-hole pairs.

        This is called implicitly by get_spectrum, get_transitions, etc.
        but there is no harm for calling this explicitly.
        """
        if not self.calc_ready:
            # Initialize wfs, paw corrections and xc, if not done yet
            # FIXME: CHECK THIS STUFF, DOES GLLB WORK???
            if not self.calc_ready:
                mode = self.calc.wfs.mode
                C_nM = self.calc.wfs.kpt_u[self.kpt_ind].C_nM
                psit_nG = self.calc.wfs.kpt_u[self.kpt_ind].psit_nG
                if ((mode == 'lcao' and C_nM is None) or
                    (mode == 'fd' and psit_nG is None)):
                    raise RuntimeError('Use ground state calculator ' +
                                       'containing the wave functions.')

                if not self.calc.scf.converged:
                    # Do not allow new scf cycles, it could change
                    # the wave function signs after every restart.
                    raise RuntimeError('Converge wave functions first.')

                spos_ac = self.calc.initialize_positions()
                self.calc.occupations.calculate(self.calc.wfs)
                self.calc.wfs.initialize(self.calc.density,
                                         self.calc.hamiltonian, spos_ac)
                self.xc.initialize(self.calc.density, self.calc.hamiltonian,
                                   self.calc.wfs, self.calc.occupations)
                if mode == 'lcao':
                    self.calc.wfs.initialize_wave_functions_from_lcao()
                self.calc_ready = True

        # Singles logic
        if self.recalculate == 'all' or self.recalculate == 'matrix':
            self.ks_singles.update_list()
            self.ks_singles.calculate()
        elif self.recalculate == 'eigen':
            self.ks_singles.read()
            self.ks_singles.kss_list_ready = True
            self.ks_singles.kss_prop_ready = True
        elif ( ( not self.ks_singles.kss_list_ready ) or
               ( not self.ks_singles.kss_prop_ready ) ):
            self.ks_singles.read()
            self.ks_singles.update_list()
            self.ks_singles.calculate()
        else:
            pass

        # K-matrix logic
        if self.recalculate == 'all' or self.recalculate == 'matrix':
            # delete files
            if self.lr_comms.parent_comm.rank == 0:
                for ready_file in glob.glob(self.basefilename +
                                            '.ready_rows.*'):
                    os.remove(ready_file)
                for K_file in glob.glob(self.basefilename + '.K_matrix.*'):
                    os.remove(K_file)
            self.K_matrix.calculate()
        elif self.recalculate == 'eigen':
            self.K_matrix.read_indices()
            self.K_matrix.K_matrix_ready = True
        elif not self.K_matrix.K_matrix_ready:
            self.K_matrix.read_indices()
            self.K_matrix.calculate()
        else:
            pass

        # Wait... we don't want to read incomplete files
        self.lr_comms.parent_comm.barrier()

        if not self.K_matrix_values_ready:
            self.K_matrix.read_values()

        # lr_transitions logic
        if not self.lr_transitions.trans_prop_ready:
            trans_file = self.basefilename + '.transitions'
            if os.path.exists(trans_file) and os.path.isfile(trans_file):
                os.remove(trans_file)
            self.lr_transitions.calculate()

        # recalculate only once
        self.recalculate = None

    def read(self, basename):
        """Does not do much at the moment."""
        info_file = open(basename + '.lr_info', 'r')
        for line in info_file:
            if line[0] == '#':
                continue
            if len(line.split()) <= 2:
                continue
            # key = line.split('=')[0]
            # value = line.split('=')[1]
            # .....
            # FIXME: do something, like warn if changed
            # ...
        info_file.close()

    def write_info(self, basename):
        """Writes used parameters to a file."""
        f = open(basename + '.lr_info', 'a+')
        f.write('# LrTDDFTindexed\n')
        f.write('%20s = %s\n' % ('xc_name', self.xc_name))
        f.write('%20s = %d\n' % ('min_occ', self.min_occ))
        f.write('%20s = %d\n' % ('min_unocc', self.min_unocc))
        f.write('%20s = %d\n' % ('max_occ', self.max_occ))
        f.write('%20s = %d\n' % ('max_unocc', self.max_unocc))
        f.write('%20s = %18.12lf\n' % ('max_energy_diff',self.max_energy_diff))
        f.write('%20s = %18.12lf\n' % ('deriv_scale', self.deriv_scale))
        f.write('%20s = %18.12lf\n' % ('min_pop_diff', self.min_pop_diff))
        f.close()

    def __del__(self):
        self.timer.stop('LrTDDFT')
