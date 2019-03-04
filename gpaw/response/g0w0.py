from __future__ import division, print_function

import functools
import itertools
import os
import pickle
import warnings
from math import pi

import numpy as np
from ase.dft.kpoints import monkhorst_pack
from ase.units import Ha
from ase.utils import opencew, devnull, pickleload
from ase.utils.timing import timer
from ase.parallel import paropen

from gpaw import GPAW
import gpaw.mpi as mpi
from gpaw import debug
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.response.chi0 import Chi0, HilbertTransform
from gpaw.response.pair import PairDensity
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb
from gpaw.response.kernels import get_coulomb_kernel
from gpaw.response.kernels import get_integrated_kernel
from gpaw.response.fxckernel_calc import calculate_kernel
from gpaw.xc.fxc import set_flags
from gpaw.wavefunctions.pw import (PWDescriptor, count_reciprocal_vectors,
                                   PWMapping)
from gpaw.xc.exx import EXX, select_kpts
from gpaw.xc.tools import vxc
from gpaw.utilities.progressbar import ProgressBar


class G0W0(PairDensity):
    def __init__(self, calc, filename='gw', restartfile=None,
                 kpts=None, bands=None, relbands=None, nbands=None, ppa=False,
                 xc='RPA', fxc_mode='GW', density_cut=1.e-6, do_GW_too=False,
                 av_scheme=None, Eg=None,
                 truncation=None, integrate_gamma=0,
                 ecut=150.0, eta=0.1, E0=1.0 * Ha,
                 domega0=0.025, omega2=10.0, q0_correction=False,
                 anisotropy_correction=None,
                 nblocks=1, savew=False, savepckl=True,
                 maxiter=1, method='G0W0', mixing=0.2,
                 world=mpi.world, ecut_extrapolation=False,
                 nblocksmax=False, gate_voltage=None):

        """G0W0 calculator.

        The G0W0 calculator is used is used to calculate the quasi
        particle energies through the G0W0 approximation for a number
        of states.

        calc: str or PAW object
            GPAW calculator object or filename of saved calculator object.
        filename: str
            Base filename of output files.
        restartfile: str
            File that stores data necessary to restart a calculation.
        kpts: list
            List of indices of the IBZ k-points to calculate the quasi particle
            energies for.
        bands: tuple of two ints
            Range of band indices, like (n1, n2), to calculate the quasi
            particle energies for. Bands n where n1<=n<n2 will be
            calculated.  Note that the second band index is not included.
        relbands: tuple of two ints
            Same as *bands* except that the numbers are relative to the
            number of occupied bands.
            E.g. (-1, 1) will use HOMO+LUMO.
        ecut: float
            Plane wave cut-off energy in eV.
        ecut_extrapolation: bool or array
            If set to True an automatic extrapolation of the selfenergy to
            infinite cutoff will be performed based on three points
            for the cutoff energy.
            If an array is given, the extrapolation will be performed based on
            the cutoff energies given in the array.
        nbands: int
            Number of bands to use in the calculation. If None, the number will
            be determined from :ecut: to yield a number close to the number of
            plane waves used.
        ppa: bool
            Sets whether the Godby-Needs plasmon-pole approximation for the
            dielectric function should be used.
        xc: str
            Kernel to use when including vertex corrections.
        fxc_mode: str
            Where to include the vertex corrections; polarizability and/or
            self-energy. 'GWP': Polarizability only, 'GWS': Self-energy only,
            'GWG': Both.
        density_cut: float
            Cutoff for density when constructing kernel.
        do_GW_too: bool
            When carrying out a calculation including vertex corrections, it
            is possible to get the standard GW results at the same time
            (almost for free).
        av_scheme: str
            'wavevector'. Method to construct kernel. Only
            'wavevector' has been tested and works here. The implementation
            could be extended to include the 'density' method which has been
            tested for total energy calculations (rALDA etc.)
        Eg: float
            Gap to apply in the 'JGMs' (simplified jellium-with-gap) kernel.
            If None the DFT gap is used.
        truncation: str
            Coulomb truncation scheme. Can be either wigner-seitz,
            2D, 1D, or 0D
        integrate_gamma: int
            Method to integrate the Coulomb interaction. 1 is a numerical
            integration at all q-points with G=[0,0,0] - this breaks the
            symmetry slightly. 0 is analytical integration at q=[0,0,0] only -
            this conserves the symmetry. integrate_gamma=2 is the same as 1,
            but the average is only carried out in the non-periodic directions.
        E0: float
            Energy (in eV) used for fitting in the plasmon-pole approximation.
        domega0: float
            Minimum frequency step (in eV) used in the generation of the non-
            linear frequency grid.
        omega2: float
            Control parameter for the non-linear frequency grid, equal to the
            frequency where the grid spacing has doubled in size.
        gate_voltage: float
            Shift Fermi level of ground state calculation by the
            specified amount.
        q0_correction: bool
            Analytic correction to the q=0 contribution applicable to 2D
            systems.
        anisotropy_correction: bool
            Old term for the q0_correction.
        nblocks: int
            Number of blocks chi0 should be distributed in so each core
            does not have to store the entire matrix. This is to reduce
            memory requirement. nblocks must be less than or equal to the
            number of processors.
        nblocksmax: bool
            Cuts chi0 into as many blocks as possible to reduce memory
            requirements as much as possible.
        savew: bool
            Save W to a file.
        savepckl: bool
            Save output to a pckl file.
        method: str
            G0W0 or GW0(eigenvalue selfconsistency in G) currently available.
        maxiter: int
            Number of iterations in a GW0 calculation.
        mixing: float
            Number between 0 and 1 determining how much of previous
            iteration's eigenvalues to mix with.
        ecut_extrapolation: bool
            Carries out the extrapolation to infinite cutoff automatically.
        """

        if world.rank != 0:
            txt = devnull
        else:
            txt = open(filename + '.txt', 'w')

        p = functools.partial(print, file=txt)
        p('  ___  _ _ _ ')
        p(' |   || | | |')
        p(' | | || | | |')
        p(' |__ ||_____|')
        p(' |___|')
        p()

        self.inputcalc = calc

        if ppa and (nblocks > 1 or nblocksmax):
            p('PPA is currently not compatible with block parallellisation. '
              'Setting nblocks=1 and continuing.')
            nblocks = 1
            nblocksmax = False

        if ecut_extrapolation is True:
            pct = 0.8
            necuts = 3
            ecut_e = ecut * (1 + (1. / pct - 1) * np.arange(necuts)[::-1] /
                             (necuts - 1))**(-2 / 3)
            """It doesn't make sence to save W in this case since
            W is calculated for different cutoffs:"""
            assert not savew
        elif isinstance(ecut_extrapolation, (list, np.ndarray)):
            ecut_e = np.array(np.sort(ecut_extrapolation))
            ecut = ecut_e[-1]
            assert not savew
        else:
            ecut_e = np.array([ecut])

        # Check if nblocks is compatible, adjust if not
        if nblocksmax:
            nblocks = world.size
            if isinstance(calc, GPAW):
                raise Exception('Using a calulator is not implemented at '
                                'the moment, load from file!')
                # nblocks_calc = calc
            else:
                nblocks_calc = GPAW(calc, txt=None)
            ngmax = []
            for q_c in nblocks_calc.wfs.kd.bzk_kc:
                qd = KPointDescriptor([q_c])
                pd = PWDescriptor(np.min(ecut) / Ha,
                                  nblocks_calc.wfs.gd, complex, qd)
                ngmax.append(pd.ngmax)
            nG = np.min(ngmax)

            while nblocks > nG**0.5 + 1 or world.size % nblocks != 0:
                nblocks -= 1

            mynG = (nG + nblocks - 1) // nblocks
            assert mynG * (nblocks - 1) < nG

        self.ecut_e = ecut_e / Ha

        PairDensity.__init__(self, calc, ecut, world=world, nblocks=nblocks,
                             gate_voltage=gate_voltage, txt=txt)

        self.gate_voltage = gate_voltage
        ecut /= Ha

        self.xc = xc
        self.density_cut = density_cut
        self.av_scheme = av_scheme
        self.fxc_mode = fxc_mode
        self.do_GW_too = do_GW_too

        if self.av_scheme is not None:
            assert self.av_scheme == 'wavevector'

        if not self.fxc_mode == 'GW':
            assert self.xc != 'RPA'

        if self.do_GW_too:
            assert self.xc != 'RPA'
            assert self.fxc_mode != 'GW'
            if restartfile is not None:
                raise RuntimeError('Restart function does not currently work '
                                   'with do_GW_too=True.')

        if Eg is None and self.xc == 'JGMsx':
            from ase.dft.bandgap import get_band_gap
            gap, k1, k2 = get_band_gap(self.calc)
            self.Eg = gap
        else:
            self.Eg = Eg

        set_flags(self)

        self.filename = filename
        self.restartfile = restartfile
        self.savew = savew
        self.savepckl = savepckl
        self.ppa = ppa
        self.truncation = truncation
        self.integrate_gamma = integrate_gamma
        self.eta = eta / Ha
        self.E0 = E0 / Ha
        self.domega0 = domega0 / Ha
        self.omega2 = omega2 / Ha
        if anisotropy_correction is not None:
            self.ac = anisotropy_correction
            warnings.warn('anisotropy_correction changed name to '
                          'q0_correction. Please update your script(s).')
        else:
            self.ac = q0_correction

        if self.ac:
            assert self.truncation == '2D'
            self.x0density = 0.1  # ? 0.01

        self.maxiter = maxiter
        self.method = method
        self.mixing = mixing
        if self.method == 'GW0':
            assert self.maxiter > 1

        self.kpts = list(select_kpts(kpts, self.calc))

        if bands is not None and relbands is not None:
            raise ValueError('Use bands or relbands!')

        if relbands is not None:
            bands = [self.calc.wfs.nvalence // 2 + b for b in relbands]

        if bands is None:
            bands = [0, self.nocc2]

        self.bands = bands

        self.ibzk_kc = self.calc.get_ibz_k_points()
        nibzk = len(self.ibzk_kc)
        self.eps0_skn = np.array([[self.calc.get_eigenvalues(kpt=k, spin=s)
                                   for k in range(nibzk)]
                                  for s in range(self.calc.wfs.nspins)]) / Ha

        b1, b2 = bands
        self.shape = shape = (self.calc.wfs.nspins, len(self.kpts), b2 - b1)
        self.eps_skn = np.empty(shape)     # KS-eigenvalues
        self.f_skn = np.empty(shape)       # occupation numbers
        self.sigma_skn = np.zeros(shape)   # self-energies
        self.dsigma_skn = np.zeros(shape)  # derivatives of self-energies
        self.vxc_skn = None                # KS XC-contributions
        self.exx_skn = None                # exact exchange contributions
        self.Z_skn = None                  # renormalization factors

        if nbands is None:
            nbands = int(self.vol * ecut**1.5 * 2**0.5 / 3 / pi**2)
        self.nbands = nbands
        self.nspins = self.calc.wfs.nspins

        if self.nspins != 1 and self.fxc_mode != 'GW':
            raise RuntimeError('Including a xc kernel does currently not '
                               'work for spinpolarized systems.')

        p()
        p('Quasi particle states:')
        if kpts is None:
            p('All k-points in IBZ')
        else:
            kptstxt = ', '.join(['{0:d}'.format(k) for k in self.kpts])
            p('k-points (IBZ indices): [' + kptstxt + ']')
        p('Band range: ({0:d}, {1:d})'.format(b1, b2))
        p()
        p('Computational parameters:')
        if not ecut_extrapolation:
            p('Plane wave cut-off: {0:g} eV'.format(self.ecut * Ha))
        else:
            p('Extrapolating to infinite plane wave cut-off using points at:')
            p('  [%.3f, %.3f, %.3f] eV' % tuple(self.ecut_e[:] * Ha))
        p('Number of bands: {0:d}'.format(self.nbands))
        p('Coulomb cutoff:', self.truncation)
        p('Broadening: {0:g} eV'.format(self.eta * Ha))
        p()
        p('Self-consistency method:', self.method)
        p('fxc mode:', self.fxc_mode)
        p('Kernel:', self.xc)
        p('Averaging scheme:', self.av_scheme)
        p('Do GW too:', self.do_GW_too)

        if self.method == 'GW0':
            p('Number of iterations:', self.maxiter)
            p('Mixing:', self.mixing)
        p()
        kd = self.calc.wfs.kd

        self.mysKn1n2 = None  # my (s, K, n1, n2) indices
        self.distribute_k_points_and_bands(b1, b2, kd.ibz2bz_k[self.kpts])

        # Find q-vectors and weights in the IBZ:
        assert -1 not in kd.bz2bz_ks
        offset_c = 0.5 * ((kd.N_c + 1) % 2) / kd.N_c
        bzq_qc = monkhorst_pack(kd.N_c) + offset_c
        self.qd = KPointDescriptor(bzq_qc)
        self.qd.set_symmetry(self.calc.atoms, kd.symmetry)

        txt.flush()

    @timer('G0W0')
    def calculate(self):
        """Starts the G0W0 calculation.

        Returns a dict with the results with the following key/value pairs:

        ===========  =============================================
        key          value
        ===========  =============================================
        ``f``        Occupation numbers
        ``eps``      Kohn-Sham eigenvalues in eV
        ``vxc``      Exchange-correlation
                     contributions in eV
        ``exx``      Exact exchange contributions in eV
        ``sigma``    Self-energy contributions in eV
        ``dsigma``   Self-energy derivatives
        ``sigma_e``  Self-energy contributions in eV
                     used for ecut extrapolation
        ``Z``        Renormalization factors
        ``qp``       Quasi particle (QP) energies in eV
        ``iqp``      GW0/GW: QP energies for each iteration in eV
        ===========  =============================================

        All the values are ``ndarray``'s of shape
        (spins, IBZ k-points, bands)."""

        kd = self.calc.wfs.kd

        self.calculate_ks_xc_contribution()
        self.calculate_exact_exchange()

        if self.restartfile is not None:
            loaded = self.load_restart_file()
            if not loaded:
                self.last_q = -1
                self.previous_sigma = 0.
                self.previous_dsigma = 0.

            else:
                print('Reading ' + str(self.last_q + 1) +
                      ' q-point(s) from the previous calculation: ' +
                      self.restartfile + '.sigma.pckl', file=self.fd)
        else:
            self.last_q = -1
            self.previous_sigma = 0.
            self.previous_dsigma = 0.

        self.fd.flush()

        self.ite = 0

        while self.ite < self.maxiter:
            # Reset calculation
            # self-energies
            self.sigma_eskn = np.zeros((len(self.ecut_e), ) + self.shape)
            # derivatives of self-energies
            self.dsigma_eskn = np.zeros((len(self.ecut_e), ) + self.shape)

            if self.do_GW_too:
                self.sigma_GW_eskn = np.zeros((len(self.ecut_e), ) +
                                              self.shape)
                self.dsigma_GW_eskn = np.zeros((len(self.ecut_e), ) +
                                               self.shape)

            # Get KS eigenvalues and occupation numbers:
            if self.ite == 0:
                b1, b2 = self.bands
                nibzk = self.calc.wfs.kd.nibzkpts
                for i, k in enumerate(self.kpts):
                    for s in range(self.nspins):
                        u = s * nibzk + k
                        kpt = self.calc.wfs.kpt_u[u]
                        self.eps_skn[s, i] = kpt.eps_n[b1:b2]
                        self.f_skn[s, i] = kpt.f_n[b1:b2] / kpt.weight

                self.qp_skn = self.eps_skn.copy()
                self.qp_iskn = np.array([self.qp_skn])
                if self.do_GW_too:
                    self.qp_GW_skn = self.eps_skn.copy()
                    self.qp_GW_iskn = np.array([self.qp_GW_skn])

            if self.ite > 0:
                self.update_energies(mixing=self.mixing)

            # My part of the states we want to calculate QP-energies for:
            mykpts = [self.get_k_point(s, K, n1, n2)
                      for s, K, n1, n2 in self.mysKn1n2]
            nkpt = len(mykpts)

            # Loop over q in the IBZ:
            nQ = 0
            for ie, pd0, W0, q_c, m2, W0_GW in \
                    self.calculate_screened_potential():
                if nQ == 0:
                    print('Summing all q:', file=self.fd)
                    pb = ProgressBar(self.fd)
                for u, kpt1 in enumerate(mykpts):
                    pb.update((nQ + 1) * u / nkpt / len(self.qd))
                    K2 = kd.find_k_plus_q(q_c, [kpt1.K])[0]
                    kpt2 = self.get_k_point(kpt1.s, K2, 0, m2,
                                            block=True)
                    k1 = kd.bz2ibz_k[kpt1.K]
                    i = self.kpts.index(k1)

                    self.calculate_q(ie, i, kpt1, kpt2, pd0, W0, W0_GW)
                nQ += 1
            pb.finish()

            self.world.sum(self.sigma_eskn)
            self.world.sum(self.dsigma_eskn)

            if self.do_GW_too:
                self.world.sum(self.sigma_GW_eskn)
                self.world.sum(self.dsigma_GW_eskn)

            if self.restartfile is not None and loaded:
                self.sigma_eskn += self.previous_sigma
                self.dsigma_eskn += self.previous_dsigma

            if len(self.ecut_e) > 1:  # interpolate to infinite ecut
                self.extrapolate_ecut()
            else:
                self.sigma_skn = self.sigma_eskn[0]
                self.dsigma_skn = self.dsigma_eskn[0]
                if self.do_GW_too:
                    self.sigma_GW_skn = self.sigma_GW_eskn[0]
                    self.dsigma_GW_skn = self.dsigma_GW_eskn[0]

            self.Z_skn = 1 / (1 - self.dsigma_skn)

            qp_skn = self.qp_skn + self.Z_skn * (
                self.eps_skn -
                self.vxc_skn - self.qp_skn + self.exx_skn +
                self.sigma_skn)

            self.qp_skn = qp_skn

            self.qp_iskn = np.concatenate((self.qp_iskn,
                                           np.array([self.qp_skn])))

            if self.do_GW_too:
                self.Z_GW_skn = 1 / (1 - self.dsigma_GW_skn)

                qp_GW_skn = self.qp_GW_skn + self.Z_GW_skn * (
                    self.eps_skn -
                    self.vxc_skn - self.qp_GW_skn + self.exx_skn +
                    self.sigma_GW_skn)

                self.qp_GW_skn = qp_GW_skn

            self.ite += 1

        results = {'f': self.f_skn,
                   'eps': self.eps_skn * Ha,
                   'vxc': self.vxc_skn * Ha,
                   'exx': self.exx_skn * Ha,
                   'sigma': self.sigma_skn * Ha,
                   'dsigma': self.dsigma_skn,
                   'Z': self.Z_skn,
                   'qp': self.qp_skn * Ha,
                   'iqp': self.qp_iskn * Ha}

        if self.do_GW_too:
            self.results_GW = {'f': self.f_skn,
                               'eps': self.eps_skn * Ha,
                               'vxc': self.vxc_skn * Ha,
                               'exx': self.exx_skn * Ha,
                               'sigma': self.sigma_GW_skn * Ha,
                               'dsigma': self.dsigma_GW_skn,
                               'Z': self.Z_GW_skn,
                               'qp': self.qp_GW_skn * Ha,
                               'iqp': self.qp_GW_iskn * Ha}

        self.print_results(results)

        if len(self.ecut_e) > 1:
            # save non-extrapolated result and R^2 value for fit quality.
            results.update({'sigma_eskn': self.sigma_eskn * Ha,
                            'dsigma_eskn': self.dsigma_eskn * Ha,
                            'sigr2_skn': self.sigr2_skn,
                            'dsigr2_skn': self.dsigr2_skn})

            if self.do_GW_too:
                self.results_GW.update(
                    {'sigma_GW_eskn': self.sigma_GW_eskn * Ha,
                     'dsigma_GW_eskn': self.dsigma_GW_eskn * Ha,
                     'sigr2_GW_skn': self.sigr2_GW_skn,
                     'dsigr2_GW_skn': self.dsigr2_GW_skn})

        if self.savepckl:
            with paropen(self.filename + '_results.pckl', 'wb') as fd:
                pickle.dump(results, fd, 2)
            if self.do_GW_too:
                with paropen(self.filename + '_results_GW.pckl', 'wb') as fd:
                    pickle.dump(self.results_GW, fd, 2)

        # After we have written the results restartfile is obsolete
        if self.restartfile is not None:
            if self.world.rank == 0:
                if os.path.isfile(self.restartfile + '.sigma.pckl'):
                    os.remove(self.restartfile + '.sigma.pckl')

        if self.method == 'GW0' and self.world.rank == 0:
            for iq in range(len(self.qd.ibzk_kc)):
                try:
                    os.remove(self.filename + '.rank' +
                              str(self.blockcomm.rank) + '.w.q' +
                              str(iq) + '.pckl')
                except OSError:
                    pass

        return results

    def calculate_q(self, ie, k, kpt1, kpt2, pd0, W0, W0_GW=None):
        """Calculates the contribution to the self-energy and its derivative
        for a given set of k-points, kpt1 and kpt2."""

        if W0_GW is None:
            Ws = [W0]
        else:
            Ws = [W0, W0_GW]

        wfs = self.calc.wfs

        N_c = pd0.gd.N_c
        i_cG = self.sign * np.dot(self.U_cc,
                                  np.unravel_index(pd0.Q_qG[0], N_c))

        q_c = wfs.kd.bzk_kc[kpt2.K] - wfs.kd.bzk_kc[kpt1.K]

        shift0_c = q_c - self.sign * np.dot(self.U_cc, pd0.kd.bzk_kc[0])
        assert np.allclose(shift0_c.round(), shift0_c)
        shift0_c = shift0_c.round().astype(int)

        shift_c = kpt1.shift_c - kpt2.shift_c - shift0_c
        I_G = np.ravel_multi_index(i_cG + shift_c[:, None], N_c, 'wrap')

        G_Gv = pd0.get_reciprocal_vectors()
        pos_av = np.dot(self.spos_ac, pd0.gd.cell_cv)
        M_vv = np.dot(pd0.gd.cell_cv.T,
                      np.dot(self.U_cc.T,
                             np.linalg.inv(pd0.gd.cell_cv).T))

        Q_aGii = []
        for a, Q_Gii in enumerate(self.Q_aGii):
            x_G = np.exp(1j * np.dot(G_Gv, (pos_av[a] -
                                            np.dot(M_vv, pos_av[a]))))
            U_ii = self.calc.wfs.setups[a].R_sii[self.s]
            Q_Gii = np.dot(np.dot(U_ii, Q_Gii * x_G[:, None, None]),
                           U_ii.T).transpose(1, 0, 2)
            if self.sign == -1:
                Q_Gii = Q_Gii.conj()
            Q_aGii.append(Q_Gii)

        if debug:
            self.check(ie, i_cG, shift0_c, N_c, q_c, Q_aGii)

        if self.ppa:
            calculate_sigma = self.calculate_sigma_ppa
        else:
            calculate_sigma = self.calculate_sigma

        for n in range(kpt1.n2 - kpt1.n1):
            ut1cc_R = kpt1.ut_nR[n].conj()
            eps1 = kpt1.eps_n[n]
            C1_aGi = [np.dot(Qa_Gii, P1_ni[n].conj())
                      for Qa_Gii, P1_ni in zip(Q_aGii, kpt1.P_ani)]
            n_mG = self.calculate_pair_densities(ut1cc_R, C1_aGi, kpt2,
                                                 pd0, I_G)
            if self.sign == 1:
                n_mG = n_mG.conj()

            f_m = kpt2.f_n
            deps_m = eps1 - kpt2.eps_n

            nn = kpt1.n1 + n - self.bands[0]

            for jj, W in enumerate(Ws):
                sigma, dsigma = calculate_sigma(n_mG, deps_m, f_m, W)
                if jj == 0:
                    self.sigma_eskn[ie, kpt1.s, k, nn] += sigma
                    self.dsigma_eskn[ie, kpt1.s, k, nn] += dsigma
                else:
                    self.sigma_GW_eskn[ie, kpt1.s, k, nn] += sigma
                    self.dsigma_GW_eskn[ie, kpt1.s, k, nn] += dsigma

    def check(self, ie, i_cG, shift0_c, N_c, q_c, Q_aGii):
        I0_G = np.ravel_multi_index(i_cG - shift0_c[:, None], N_c, 'wrap')
        qd1 = KPointDescriptor([q_c])
        pd1 = PWDescriptor(self.ecut_e[ie], self.calc.wfs.gd, complex, qd1)
        G_I = np.empty(N_c.prod(), int)
        G_I[:] = -1
        I1_G = pd1.Q_qG[0]
        G_I[I1_G] = np.arange(len(I0_G))
        G_G = G_I[I0_G]
        assert len(I0_G) == len(I1_G)
        assert (G_G >= 0).all()
        for a, Q_Gii in enumerate(self.initialize_paw_corrections(pd1)):
            e = abs(Q_aGii[a] - Q_Gii[G_G]).max()
            assert e < 1e-12

    @timer('Sigma')
    def calculate_sigma(self, n_mG, deps_m, f_m, C_swGG):
        """Calculates a contribution to the self-energy and its derivative for
        a given (k, k-q)-pair from its corresponding pair-density and
        energy."""
        o_m = abs(deps_m)
        # Add small number to avoid zeros for degenerate states:
        sgn_m = np.sign(deps_m + 1e-15)

        # Pick +i*eta or -i*eta:
        s_m = (1 + sgn_m * np.sign(0.5 - f_m)).astype(int) // 2

        beta = (2**0.5 - 1) * self.domega0 / self.omega2
        w_m = (o_m / (self.domega0 + beta * o_m)).astype(int)
        m_inb = np.where(w_m < len(self.omega_w) - 1)[0]
        o1_m = np.empty(len(o_m))
        o2_m = np.empty(len(o_m))
        o1_m[m_inb] = self.omega_w[w_m[m_inb]]
        o2_m[m_inb] = self.omega_w[w_m[m_inb] + 1]

        x = 1.0 / (self.qd.nbzkpts * 2 * pi * self.vol)
        sigma = 0.0
        dsigma = 0.0
        # Performing frequency integration
        for o, o1, o2, sgn, s, w, n_G in zip(o_m, o1_m, o2_m,
                                             sgn_m, s_m, w_m, n_mG):

            if w >= len(self.omega_w) - 1:
                continue

            C1_GG = C_swGG[s][w]
            C2_GG = C_swGG[s][w + 1]
            p = x * sgn
            myn_G = n_G[self.Ga:self.Gb]

            sigma1 = p * np.dot(np.dot(myn_G, C1_GG), n_G.conj()).imag
            sigma2 = p * np.dot(np.dot(myn_G, C2_GG), n_G.conj()).imag
            sigma += ((o - o1) * sigma2 + (o2 - o) * sigma1) / (o2 - o1)
            dsigma += sgn * (sigma2 - sigma1) / (o2 - o1)

        return sigma, dsigma

    def calculate_screened_potential(self):
        """Calculates the screened potential for each q-point in the 1st BZ.
        Since many q-points are related by symmetry, the actual calculation is
        only done for q-points in the IBZ and the rest are obtained by symmetry
        transformations. Results are returned as a generator to that it is not
        necessary to store a huge matrix for each q-point in the memory."""
        # The decorator $timer('W') doesn't work for generators, do we will
        # have to manually start and stop the timer here:
        self.timer.start('W')
        if self.ite == 0:
            print('\nCalculating screened Coulomb potential', file=self.fd)
            if self.truncation is not None:
                print('Using %s truncated Coloumb potential' % self.truncation,
                      file=self.fd)

        if self.ppa:
            if self.ite == 0:
                print('Using Godby-Needs plasmon-pole approximation:',
                      file=self.fd)
                print('  Fitting energy: i*E0, E0 = %.3f Hartee' % self.E0,
                      file=self.fd)

            # use small imaginary frequency to avoid dividing by zero:
            frequencies = [1e-10j, 1j * self.E0 * Ha]

            parameters = {'eta': 0,
                          'hilbert': False,
                          'timeordered': False,
                          'frequencies': frequencies}
        else:
            if self.ite == 0:
                print('Using full frequency integration:', file=self.fd)
                print('  domega0: {0:g}'.format(self.domega0 * Ha),
                      file=self.fd)
                print('  omega2: {0:g}'.format(self.omega2 * Ha),
                      file=self.fd)

            parameters = {'eta': self.eta * Ha,
                          'hilbert': True,
                          'timeordered': True,
                          'domega0': self.domega0 * Ha,
                          'omega2': self.omega2 * Ha}

        self.fd.flush()

        chi0 = Chi0(self.inputcalc,
                    nbands=self.nbands,
                    ecut=self.ecut * Ha,
                    intraband=False,
                    real_space_derivatives=False,
                    txt=self.filename + '.w.txt',
                    timer=self.timer,
                    nblocks=self.blockcomm.size,
                    gate_voltage=self.gate_voltage,
                    **parameters)

        if self.truncation == 'wigner-seitz':
            wstc = WignerSeitzTruncatedCoulomb(
                self.calc.wfs.gd.cell_cv,
                self.calc.wfs.kd.N_c,
                chi0.fd)
        else:
            wstc = None

        self.omega_w = chi0.omega_w
        self.omegamax = chi0.omegamax

        htp = HilbertTransform(self.omega_w, self.eta, gw=True)
        htm = HilbertTransform(self.omega_w, -self.eta, gw=True)

        # Find maximum size of chi-0 matrices:
        gd = self.calc.wfs.gd
        nGmax = max(count_reciprocal_vectors(self.ecut, gd, q_c)
                    for q_c in self.qd.ibzk_kc)
        nw = len(self.omega_w)

        size = self.blockcomm.size

        mynGmax = (nGmax + size - 1) // size
        mynw = (nw + size - 1) // size

        # some memory sizes...
        if self.world.rank == 0:
            # A1_x, A2_x
            siz = (nw * mynGmax * nGmax +
                   max(mynw * nGmax, nw * mynGmax) * nGmax) * 16
            sizA = (nw * nGmax * nGmax + nw * nGmax * nGmax) * 16
            print('  memory estimate for chi0: local=%.2f MB, global=%.2f MB'
                  % (siz / 1024**2, sizA / 1024**2), file=self.fd)
            self.fd.flush()

        # Allocate memory in the beginning and use for all q:
        A1_x = np.empty(nw * mynGmax * nGmax, complex)
        A2_x = np.empty(max(mynw * nGmax, nw * mynGmax) * nGmax, complex)

        # Need to pause the timer in between iterations
        self.timer.stop('W')
        for iq, q_c in enumerate(self.qd.ibzk_kc):
            if iq <= self.last_q:
                continue

            thisqd = KPointDescriptor([q_c])
            pd = PWDescriptor(self.ecut, self.calc.wfs.gd, complex, thisqd)
            nG = pd.ngmax
            mynG = (nG + self.blockcomm.size - 1) // self.blockcomm.size
            chi0.Ga = self.blockcomm.rank * mynG
            chi0.Gb = min(chi0.Ga + mynG, nG)
            if len(self.ecut_e) > 1:
                shape = (nw, chi0.Gb - chi0.Ga, nG)
                chi0bands_wGG = A1_x[:np.prod(shape)].reshape(shape).copy()
                chi0bands_wGG[:] = 0.0

                if np.allclose(q_c, 0.0):
                    chi0bands_wxvG = np.zeros((nw, 2, 3, nG), complex)
                    chi0bands_wvv = np.zeros((nw, 3, 3), complex)
                else:
                    chi0bands_wxvG = None
                    chi0bands_wvv = None
            else:
                chi0bands_wGG = None
                chi0bands_wxvG = None
                chi0bands_wvv = None

            m1 = chi0.nocc1
            for ie, ecut in enumerate(self.ecut_e):
                self.timer.start('W')
                if self.savew:
                    wfilename = (self.filename + '.rank' +
                                 str(self.blockcomm.rank) +
                                 '.w.q%d.pckl' % iq)
                    fd = opencew(wfilename)
                if self.savew and fd is None:
                    # Read screened potential from file
                    with open(wfilename, 'rb') as fd:
                        pdi, W = pickleload(fd)
                    # We also need to initialize the PAW corrections
                    self.Q_aGii = self.initialize_paw_corrections(pdi)

                    nG = pdi.ngmax
                    nw = len(self.omega_w)
                    mynG = (nG + self.blockcomm.size - 1) // \
                        self.blockcomm.size
                    self.Ga = self.blockcomm.rank * mynG
                    self.Gb = min(self.Ga + mynG, nG)
                    assert mynG * (self.blockcomm.size - 1) < nG
                else:
                    # First time calculation
                    if ecut == self.ecut:
                        # Nothing to cut away:
                        m2 = self.nbands
                    else:
                        m2 = int(self.vol * ecut**1.5 * 2**0.5 / 3 / pi**2)

                    pdi, W, W_GW = self.calculate_w(
                        chi0, q_c, pd, chi0bands_wGG,
                        chi0bands_wxvG, chi0bands_wvv,
                        m1, m2, ecut, htp, htm, wstc,
                        A1_x, A2_x, iq)
                    m1 = m2
                    if self.savew:
                        if self.blockcomm.size > 1:
                            thisfile = (self.filename + '.rank' +
                                        str(self.blockcomm.rank) +
                                        '.w.q%d.pckl' % iq)
                            with open(thisfile, 'wb') as fd:
                                pickle.dump((pdi, W), fd, 2)
                        else:
                            pickle.dump((pdi, W), fd, 2)

                self.timer.stop('W')
                # Loop over all k-points in the BZ and find those that are
                # related to the current IBZ k-point by symmetry
                Q1 = self.qd.ibz2bz_k[iq]
                done = set()
                for Q2 in self.qd.bz2bz_ks[Q1]:
                    if Q2 >= 0 and Q2 not in done:
                        s = self.qd.sym_k[Q2]
                        self.s = s
                        self.U_cc = self.qd.symmetry.op_scc[s]
                        time_reversal = self.qd.time_reversal_k[Q2]
                        self.sign = 1 - 2 * time_reversal
                        Q_c = self.qd.bzk_kc[Q2]
                        d_c = self.sign * np.dot(self.U_cc, q_c) - Q_c
                        assert np.allclose(d_c.round(), d_c)
                        yield ie, pdi, W, Q_c, m2, W_GW
                        done.add(Q2)

                if self.restartfile is not None:
                    self.save_restart_file(iq)

    @timer('WW')
    def calculate_w(self, chi0, q_c, pd, chi0bands_wGG, chi0bands_wxvG,
                    chi0bands_wvv, m1, m2, ecut, htp, htm, wstc, A1_x, A2_x,
                    iq):
        """Calculates the screened potential for a specified q-point."""

        nw = len(self.omega_w)
        nG = pd.ngmax
        mynG = (nG + self.blockcomm.size - 1) // self.blockcomm.size
        self.Ga = chi0.Ga
        self.Gb = chi0.Gb
        shape = (nw, chi0.Gb - chi0.Ga, nG)
        # construct empty matrix for chi
        chi0_wGG = A1_x[:np.prod(shape)].reshape(shape).copy()
        chi0_wGG[:] = 0.0
        if np.allclose(q_c, 0.0):
            chi0_wxvG = np.zeros((nw, 2, 3, nG), complex)
            chi0_wvv = np.zeros((nw, 3, 3), complex)
        else:
            chi0_wxvG = None
            chi0_wvv = None

        chi0._calculate(pd, chi0_wGG, chi0_wxvG, chi0_wvv, m1, m2,
                        range(self.nspins), extend_head=False)

        if len(self.ecut_e) > 1:
            # Add chi from previous cutoff with remaining bands
            chi0_wGG += chi0bands_wGG
            chi0bands_wGG[:] = chi0_wGG.copy()
            if np.allclose(q_c, 0.0):
                chi0_wxvG += chi0bands_wxvG
                chi0bands_wxvG[:] = chi0_wxvG.copy()
                chi0_wvv += chi0bands_wvv
                chi0bands_wvv[:] = chi0_wvv.copy()

        self.Q_aGii = chi0.Q_aGii

        mynw = (nw + self.blockcomm.size - 1) // self.blockcomm.size
        if self.blockcomm.size > 1:
            chi0_wGG = chi0.redistribute(chi0_wGG, A2_x)
            wa = min(self.blockcomm.rank * mynw, nw)
            wb = min(wa + mynw, nw)
        else:
            wa = 0
            wb = nw

        if ecut == pd.ecut:
            nG = len(chi0_wGG[0])
            pdi = pd
            G2G = None

        elif ecut < pd.ecut:  # construct subset chi0 matrix with lower ecut
            pdi = PWDescriptor(ecut, pd.gd, dtype=pd.dtype,
                               kd=pd.kd)
            nG = pdi.ngmax
            mynG = (nG + self.blockcomm.size - 1) // self.blockcomm.size
            self.Ga = self.blockcomm.rank * mynG
            self.Gb = min(self.Ga + mynG, nG)
            nw = len(self.omega_w)
            mynw = (nw + self.blockcomm.size - 1) // self.blockcomm.size

            G2G = PWMapping(pdi, pd).G2_G1
            chi0_wGG = chi0_wGG.take(G2G, axis=1).take(G2G, axis=2)

            if chi0_wxvG is not None:
                chi0_wxvG = chi0_wxvG.take(G2G, axis=3)

            if self.Q_aGii is not None:
                for a, Q_Gii in enumerate(self.Q_aGii):
                    self.Q_aGii[a] = Q_Gii.take(G2G, axis=0)

        if self.integrate_gamma != 0:
            if self.integrate_gamma == 2:
                reduced = True
            else:
                reduced = False
            V0, sqrV0 = get_integrated_kernel(pdi,
                                              self.calc.wfs.kd.N_c,
                                              truncation=self.truncation,
                                              reduced=reduced,
                                              N=100)
        elif self.integrate_gamma == 0 and np.allclose(q_c, 0):
            bzvol = (2 * np.pi)**3 / self.vol / self.qd.nbzkpts
            Rq0 = (3 * bzvol / (4 * np.pi))**(1. / 3.)
            V0 = 16 * np.pi**2 * Rq0 / bzvol
            sqrV0 = (4 * np.pi)**(1.5) * Rq0**2 / bzvol / 2
        else:
            pass

        delta_GG = np.eye(nG)

        if self.ppa:
            einv_wGG = []

        # Calculate kernel
        fv = calculate_kernel(self, nG, self.nspins, iq, G2G)[0:nG, 0:nG]
        # Generate fine grid in vicinity of gamma
        if np.allclose(q_c, 0):
            kd = self.calc.wfs.kd
            N = 4
            N_c = np.array([N, N, N])
            if self.truncation is not None:
                # Only average periodic directions if trunction is used
                N_c[np.where(kd.N_c == 1)[0]] = 1
            qf_qc = monkhorst_pack(N_c) / kd.N_c
            qf_qc *= 1.0e-6
            U_scc = kd.symmetry.op_scc
            qf_qc = kd.get_ibz_q_points(qf_qc, U_scc)[0]
            weight_q = kd.q_weights
            qf_qv = 2 * np.pi * np.dot(qf_qc, pd.gd.icell_cv)
            a_wq = np.sum([chi0_vq * qf_qv.T
                           for chi0_vq in
                           np.dot(chi0_wvv[wa:wb], qf_qv.T)], axis=1)
            a0_qwG = np.dot(qf_qv, chi0_wxvG[wa:wb, 0])
            a1_qwG = np.dot(qf_qv, chi0_wxvG[wa:wb, 1])

        self.timer.start('Dyson eq.')
        # Calculate W and store it in chi0_wGG ndarray:
        if self.do_GW_too:
            chi0_GW_wGG = chi0_wGG.copy()
        else:
            chi0_GW_wGG = [0]

        for iw, [chi0_GG, chi0_GW_GG] in enumerate(
            zip(chi0_wGG,
                itertools.cycle(chi0_GW_wGG))):
            if np.allclose(q_c, 0):
                einv_GG = np.zeros((nG, nG), complex)
                if self.do_GW_too:
                    einv_GW_GG = np.zeros((nG, nG), complex)
                for iqf in range(len(qf_qv)):
                    chi0_GG[0] = a0_qwG[iqf, iw]
                    chi0_GG[:, 0] = a1_qwG[iqf, iw]
                    chi0_GG[0, 0] = a_wq[iw, iqf]
                    if self.do_GW_too:
                        chi0_GW_GG[0] = a0_qwG[iqf, iw]
                        chi0_GW_GG[:, 0] = a1_qwG[iqf, iw]
                        chi0_GW_GG[0, 0] = a_wq[iw, iqf]
                    sqrV_G = get_coulomb_kernel(pdi,
                                                kd.N_c,
                                                truncation=self.truncation,
                                                wstc=wstc,
                                                q_v=qf_qv[iqf])**0.5

#                    chi0v_GG = chi0_GG * sqrV_G * sqrV_G[:, np.newaxis]
#                    if self.nspins == 2:
#                        chi0v = np.zeros((2 * nG, 2 * nG), dtype=complex)
#                        for s in range(self.nspins):
#                            m = s * nG
#                            n = (s + 1) * nG
#                            chi0v[m:n, m:n] = chi0v_GG
#                    else:
#                    chi0v = chi0v_GG

                    if self.fxc_mode == 'GWP':
                        e_GG = (np.eye(nG) -
                                np.dot(
                                    np.linalg.inv(
                                        np.eye(nG) -
                                        np.dot(chi0_GG *
                                               sqrV_G *
                                               sqrV_G[:, np.newaxis], fv) +
                                        chi0_GG * sqrV_G *
                                        sqrV_G[:, np.newaxis]),
                                    chi0_GG * sqrV_G *
                                    sqrV_G[:, np.newaxis]))
                    elif self.fxc_mode == 'GWS':
                        e_GG = np.dot(
                            np.linalg.inv(
                                np.eye(nG) +
                                np.dot(chi0_GG *
                                       sqrV_G *
                                       sqrV_G[:, np.newaxis], fv) -
                                chi0_GG * sqrV_G *
                                sqrV_G[:, np.newaxis]),
                            np.eye(nG) -
                            chi0_GG * sqrV_G *
                            sqrV_G[:, np.newaxis])
                    else:
                        e_GG = np.eye(nG) - np.dot(chi0_GG * sqrV_G *
                                                   sqrV_G[:, np.newaxis], fv)

                    einv_GG += np.linalg.inv(e_GG) * weight_q[iqf]

                    if self.do_GW_too:
                        e_GW_GG = (np.eye(nG) -
                                   chi0_GW_GG * sqrV_G *
                                   sqrV_G[:, np.newaxis])
                        einv_GW_GG += np.linalg.inv(e_GW_GG) * weight_q[iqf]

            else:
                sqrV_G = get_coulomb_kernel(pdi,
                                            self.calc.wfs.kd.N_c,
                                            truncation=self.truncation,
                                            wstc=wstc)**0.5
                if self.fxc_mode == 'GWP':
                    e_GG = (np.eye(nG) -
                            np.dot(
                                np.linalg.inv(
                                    np.eye(nG) -
                                    np.dot(chi0_GG * sqrV_G *
                                           sqrV_G[:, np.newaxis], fv) +
                                    chi0_GG * sqrV_G *
                                    sqrV_G[:, np.newaxis]),
                                chi0_GG * sqrV_G * sqrV_G[:, np.newaxis]))
                elif self.fxc_mode == 'GWS':
                    e_GG = np.dot(
                        np.linalg.inv(
                            np.eye(nG) +
                            np.dot(chi0_GG * sqrV_G *
                                   sqrV_G[:, np.newaxis], fv) -
                            chi0_GG * sqrV_G *
                            sqrV_G[:, np.newaxis]),
                        np.eye(nG) - chi0_GG *
                        sqrV_G * sqrV_G[:, np.newaxis])
                else:
                    e_GG = (delta_GG -
                            np.dot(chi0_GG * sqrV_G *
                                   sqrV_G[:, np.newaxis], fv))

                einv_GG = np.linalg.inv(e_GG)

                if self.do_GW_too:
                    e_GW_GG = (delta_GG -
                               chi0_GW_GG * sqrV_G * sqrV_G[:, np.newaxis])
                    einv_GW_GG = np.linalg.inv(e_GW_GG)

            if self.ppa:
                einv_wGG.append(einv_GG - delta_GG)
            else:
                W_GG = chi0_GG
                W_GG[:] = (einv_GG - delta_GG) * sqrV_G * sqrV_G[:, np.newaxis]

                if self.do_GW_too:
                    W_GW_GG = chi0_GW_GG
                    W_GW_GG[:] = ((einv_GW_GG - delta_GG) *
                                  sqrV_G * sqrV_G[:, np.newaxis])

                if self.ac and np.allclose(q_c, 0):
                    if iw == 0:
                        print_ac = True
                    else:
                        print_ac = False
                    self.add_q0_correction(pdi, W_GG, einv_GG,
                                           chi0_wxvG[wa + iw],
                                           chi0_wvv[wa + iw],
                                           sqrV_G,
                                           print_ac=print_ac)
                    if self.do_GW_too:
                        self.add_q0_correction(pdi, W_GW_GG, einv_GW_GG,
                                               chi0_wxvG[wa + iw],
                                               chi0_wvv[wa + iw],
                                               sqrV_G,
                                               print_ac=print_ac)
                elif np.allclose(q_c, 0) or self.integrate_gamma != 0:
                    W_GG[0, 0] = (einv_GG[0, 0] - 1.0) * V0
                    W_GG[0, 1:] = einv_GG[0, 1:] * sqrV_G[1:] * sqrV0
                    W_GG[1:, 0] = einv_GG[1:, 0] * sqrV0 * sqrV_G[1:]
                    if self.do_GW_too:
                        W_GW_GG[0, 0] = (einv_GW_GG[0, 0] - 1.0) * V0
                        W_GW_GG[0, 1:] = einv_GW_GG[0, 1:] * sqrV_G[1:] * sqrV0
                        W_GW_GG[1:, 0] = einv_GW_GG[1:, 0] * sqrV0 * sqrV_G[1:]
                else:
                    pass

        if self.ppa:
            omegat_GG = self.E0 * np.sqrt(einv_wGG[1] /
                                          (einv_wGG[0] - einv_wGG[1]))
            R_GG = -0.5 * omegat_GG * einv_wGG[0]
            W_GG = pi * R_GG * sqrV_G * sqrV_G[:, np.newaxis]
            if np.allclose(q_c, 0) or self.integrate_gamma != 0:
                W_GG[0, 0] = pi * R_GG[0, 0] * V0
                W_GG[0, 1:] = pi * R_GG[0, 1:] * sqrV_G[1:] * sqrV0
                W_GG[1:, 0] = pi * R_GG[1:, 0] * sqrV0 * sqrV_G[1:]

            self.timer.stop('Dyson eq.')
            return pdi, [W_GG, omegat_GG], None

        if self.do_GW_too:
            A1_GW_x = A1_x.copy()
            A2_GW_x = A2_x.copy()

        if self.blockcomm.size > 1:
            Wm_wGG = chi0.redistribute(chi0_wGG, A1_x)
        else:
            Wm_wGG = chi0_wGG

        Wp_wGG = A2_x[:Wm_wGG.size].reshape(Wm_wGG.shape)
        Wp_wGG[:] = Wm_wGG

        with self.timer('Hilbert transform'):
            htp(Wp_wGG)
            htm(Wm_wGG)
        self.timer.stop('Dyson eq.')

        if self.do_GW_too:
            if self.blockcomm.size > 1:
                Wm_GW_wGG = chi0.redistribute(chi0_GW_wGG, A1_GW_x)
            else:
                Wm_GW_wGG = chi0_GW_wGG

            Wp_GW_wGG = A2_GW_x[:Wm_GW_wGG.size].reshape(Wm_GW_wGG.shape)
            Wp_GW_wGG[:] = Wm_GW_wGG

            htp(Wp_GW_wGG)
            htm(Wm_GW_wGG)
            GW_return = [Wp_GW_wGG, Wm_GW_wGG]
        else:
            GW_return = None

        return pdi, [Wp_wGG, Wm_wGG], GW_return

    @timer('Kohn-Sham XC-contribution')
    def calculate_ks_xc_contribution(self):
        name = self.filename + '.vxc.npy'
        fd, self.vxc_skn = self.read_contribution(name)
        if self.vxc_skn is None:
            print('Calculating Kohn-Sham XC contribution', file=self.fd)
            vxc_skn = vxc(self.calc, self.calc.hamiltonian.xc) / Ha
            n1, n2 = self.bands
            self.vxc_skn = vxc_skn[:, self.kpts, n1:n2]
            np.save(fd, self.vxc_skn)

    @timer('EXX')
    def calculate_exact_exchange(self):
        name = self.filename + '.exx.npy'
        fd, self.exx_skn = self.read_contribution(name)
        if self.exx_skn is None:
            print('Calculating EXX contribution', file=self.fd)
            exx = EXX(self.calc, kpts=self.kpts, bands=self.bands,
                      txt=self.filename + '.exx.txt', timer=self.timer)
            exx.calculate()
            self.exx_skn = exx.get_eigenvalue_contributions() / Ha
            np.save(fd, self.exx_skn)

    def read_contribution(self, filename):
        fd = opencew(filename)  # create, exclusive, write
        if fd is not None:
            # File was not there: nothing to read
            return fd, None

        try:
            with open(filename, 'rb') as fd:
                x_skn = np.load(fd)
        except IOError:
            print('Removing broken file:', filename, file=self.fd)
        else:
            print('Read:', filename, file=self.fd)
            if x_skn.shape == self.shape:
                return None, x_skn
            print('Removing bad file (wrong shape of array):', filename,
                  file=self.fd)

        if self.world.rank == 0:
            os.remove(filename)

        return opencew(filename), None

    def print_results(self, results):
        description = ['f:      Occupation numbers',
                       'eps:     KS-eigenvalues [eV]',
                       'vxc:     KS vxc [eV]',
                       'exx:     Exact exchange [eV]',
                       'sigma:   Self-energies [eV]',
                       'dsigma:  Self-energy derivatives',
                       'Z:       Renormalization factors',
                       'qp:      QP-energies [eV]']

        print('\nResults:', file=self.fd)
        for line in description:
            print(line, file=self.fd)

        b1, b2 = self.bands
        names = [line.split(':', 1)[0] for line in description]
        ibzk_kc = self.calc.wfs.kd.ibzk_kc
        for s in range(self.calc.wfs.nspins):
            for i, ik in enumerate(self.kpts):
                print('\nk-point ' +
                      '{0} ({1}): ({2:.3f}, {3:.3f}, {4:.3f})'.format(
                          i, ik, *ibzk_kc[ik]) + '                ' +
                      self.fxc_mode, file=self.fd)
                print('band' +
                      ''.join('{0:>8}'.format(name) for name in names),
                      file=self.fd)
                for n in range(b2 - b1):
                    print('{0:4}'.format(n + b1) +
                          ''.join('{0:8.3f}'.format(results[name][s, i, n])
                                  for name in names),
                          file=self.fd)
                if self.do_GW_too:
                    print(' ' * 67 + 'GW', file=self.fd)
                    for n in range(b2 - b1):
                        print('{0:4}'.format(n + b1) +
                              ''.join('{0:8.3f}'
                                      .format(self.results_GW[name][s, i, n])
                                      for name in names),
                              file=self.fd)

        self.timer.write(self.fd)

    @timer('PPA-Sigma')
    def calculate_sigma_ppa(self, n_mG, deps_m, f_m, W):
        W_GG, omegat_GG = W

        sigma = 0.0
        dsigma = 0.0

        # init variables (is this necessary?)
        nG = n_mG.shape[1]
        deps_GG = np.empty((nG, nG))
        sign_GG = np.empty((nG, nG))
        x1_GG = np.empty((nG, nG))
        x2_GG = np.empty((nG, nG))
        x3_GG = np.empty((nG, nG))
        x4_GG = np.empty((nG, nG))
        x_GG = np.empty((nG, nG))
        dx_GG = np.empty((nG, nG))
        nW_G = np.empty(nG)
        for m in range(np.shape(n_mG)[0]):
            deps_GG = deps_m[m]
            sign_GG = 2 * f_m[m] - 1
            x1_GG = 1 / (deps_GG + omegat_GG - 1j * self.eta)
            x2_GG = 1 / (deps_GG - omegat_GG + 1j * self.eta)
            x3_GG = 1 / (deps_GG + omegat_GG - 1j * self.eta * sign_GG)
            x4_GG = 1 / (deps_GG - omegat_GG - 1j * self.eta * sign_GG)
            x_GG = W_GG * (sign_GG * (x1_GG - x2_GG) + x3_GG + x4_GG)
            dx_GG = W_GG * (sign_GG * (x1_GG**2 - x2_GG**2) +
                            x3_GG**2 + x4_GG**2)
            nW_G = np.dot(n_mG[m], x_GG)
            sigma += np.vdot(n_mG[m], nW_G).real
            nW_G = np.dot(n_mG[m], dx_GG)
            dsigma -= np.vdot(n_mG[m], nW_G).real

        x = 1 / (self.qd.nbzkpts * 2 * pi * self.vol)
        return x * sigma, x * dsigma

    def save_restart_file(self, nQ):
        sigma_eskn_write = self.sigma_eskn.copy()
        dsigma_eskn_write = self.dsigma_eskn.copy()
        self.world.sum(sigma_eskn_write)
        self.world.sum(dsigma_eskn_write)
        data = {'last_q': nQ,
                'sigma_eskn': sigma_eskn_write + self.previous_sigma,
                'dsigma_eskn': dsigma_eskn_write + self.previous_dsigma,
                'kpts': self.kpts,
                'bands': self.bands,
                'nbands': self.nbands,
                'ecut_e': self.ecut_e,
                'domega0': self.domega0,
                'omega2': self.omega2,
                'integrate_gamma': self.integrate_gamma}

        if self.world.rank == 0:
            with open(self.restartfile + '.sigma.pckl', 'wb') as fd:
                pickle.dump(data, fd, 2)

    def load_restart_file(self):
        try:
            with open(self.restartfile + '.sigma.pckl', 'rb') as fd:
                data = pickleload(fd)
        except IOError:
            return False
        else:
            if (data['kpts'] == self.kpts and
                data['bands'] == self.bands and
                data['nbands'] == self.nbands and
                (data['ecut_e'] == self.ecut_e).all and
                data['domega0'] == self.domega0 and
                data['omega2'] == self.omega2 and
                data['integrate_gamma'] == self.integrate_gamma):
                self.last_q = data['last_q']
                self.previous_sigma = data['sigma_eskn']
                self.previous_dsigma = data['dsigma_eskn']
                return True
            else:
                raise ValueError(
                    'Restart file not compatible with parameters used in '
                    'current calculation. Check kpts, bands, nbands, ecut, '
                    'domega0, omega2, integrate_gamma.')

    def extrapolate_ecut(self):
        # Do linear fit of selfenergy vs. inverse of number of plane waves
        # to extrapolate to infinite number of plane waves
        from scipy.stats import linregress
        print('', file=self.fd)
        print('Extrapolating selfenergy to infinite energy cutoff:',
              file=self.fd)
        print('  Performing linear fit to %d points' % len(self.ecut_e),
              file=self.fd)
        self.sigr2_skn = np.zeros(self.shape)
        self.dsigr2_skn = np.zeros(self.shape)
        self.sigma_skn = np.zeros(self.shape)
        self.dsigma_skn = np.zeros(self.shape)
        invN_i = self.ecut_e**(-3. / 2)
        for m in range(np.product(self.shape)):
            s, k, n = np.unravel_index(m, self.shape)

            slope, intercept, r_value, p_value, std_err = \
                linregress(invN_i, self.sigma_eskn[:, s, k, n])

            self.sigr2_skn[s, k, n] = r_value**2
            self.sigma_skn[s, k, n] = intercept

            slope, intercept, r_value, p_value, std_err = \
                linregress(invN_i, self.dsigma_eskn[:, s, k, n])

            self.dsigr2_skn[s, k, n] = r_value**2
            self.dsigma_skn[s, k, n] = intercept

        if np.any(self.sigr2_skn < 0.9) or np.any(self.dsigr2_skn < 0.9):
            print('  Warning: Bad quality of linear fit for some (n,k). ',
                  file=self.fd)
            print('           Higher cutoff might be necesarry.', file=self.fd)

        print('  Minimum R^2 = %1.4f. (R^2 Should be close to 1)' %
              min(np.min(self.sigr2_skn), np.min(self.dsigr2_skn)),
              file=self.fd)

        if self.do_GW_too:
            self.sigr2_GW_skn = np.zeros(self.shape)
            self.dsigr2_GW_skn = np.zeros(self.shape)
            self.sigma_GW_skn = np.zeros(self.shape)
            self.dsigma_GW_skn = np.zeros(self.shape)
            invN_i = self.ecut_e**(-3. / 2)
            for m in range(np.product(self.shape)):
                s, k, n = np.unravel_index(m, self.shape)

                slope, intercept, r_value, p_value, std_err = \
                    linregress(invN_i, self.sigma_GW_eskn[:, s, k, n])

                self.sigr2_GW_skn[s, k, n] = r_value**2
                self.sigma_GW_skn[s, k, n] = intercept

                slope, intercept, r_value, p_value, std_err = \
                    linregress(invN_i, self.dsigma_GW_eskn[:, s, k, n])

                self.dsigr2_GW_skn[s, k, n] = r_value**2
                self.dsigma_GW_skn[s, k, n] = intercept

            if np.any(self.sigr2_GW_skn < 0.9) or np.any(self.dsigr2_GW_skn <
                                                         0.9):
                print('  GW calculation. Warning: Bad quality of linear fit '
                      'for some (n,k). ',
                      file=self.fd)
                print('           Higher cutoff might be necesarry.',
                      file=self.fd)

            print('  Minimum R^2 = %1.4f. (R^2 Should be close to 1)' %
                  min(np.min(self.sigr2_GW_skn), np.min(self.dsigr2_GW_skn)),
                  file=self.fd)

    def add_q0_correction(self, pd, W_GG, einv_GG, chi0_xvG, chi0_vv,
                          sqrV_G, print_ac=False):
        from ase.dft import monkhorst_pack
        self.cell_cv = self.calc.wfs.gd.cell_cv
        self.qpts_qc = self.calc.wfs.kd.bzk_kc
        self.weight_q = 1.0 * np.ones(len(self.qpts_qc)) / len(self.qpts_qc)
        L = self.cell_cv[2, 2]
        vc_G0 = sqrV_G[1:]**2

        B_GG = einv_GG[1:, 1:]
        u_v0G = vc_G0[np.newaxis, :]**0.5 * chi0_xvG[0, :, 1:]
        u_vG0 = vc_G0[np.newaxis, :]**0.5 * chi0_xvG[1, :, 1:]
        U_vv = -chi0_vv
        a_v0G = -np.dot(u_v0G, B_GG)
        a_vG0 = -np.dot(u_vG0, B_GG.T)
        A_vv = U_vv - np.dot(a_v0G, u_vG0.T)
        S_v0G = a_v0G
        S_vG0 = a_vG0
        L_vv = A_vv

        # Get necessary G vectors.
        G_Gv = pd.get_reciprocal_vectors()[1:]
        G_Gv += np.array([1e-14, 1e-14, 0])
        G2_G = np.sum(G_Gv**2, axis=1)
        Gpar_G = np.sum(G_Gv[:, 0:2]**2, axis=1)**0.5

        # Generate numerical q-point grid
        rcell_cv = 2 * pi * np.linalg.inv(self.cell_cv).T
        N_c = self.qd.N_c

        iq = np.argmin(np.sum(self.qpts_qc**2, axis=1))
        assert np.allclose(self.qpts_qc[iq], 0)
        q0cell_cv = np.array([1, 1, 1])**0.5 * rcell_cv / N_c
        q0vol = abs(np.linalg.det(q0cell_cv))

        x0density = self.x0density
        q0density = 2. / L * x0density
        npts_c = np.ceil(np.sum(q0cell_cv**2, axis=1)**0.5 /
                         q0density).astype(int)
        npts_c[2] = 1
        npts_c += (npts_c + 1) % 2
        if print_ac:
            print('Applying analytical 2D correction to W:',
                  file=self.fd)
            print('    Evaluating Gamma point contribution to W on a ' +
                  '%dx%dx%d grid' % tuple(npts_c), file=self.fd)

        qpts_qc = monkhorst_pack(npts_c)
        qgamma = np.argmin(np.sum(qpts_qc**2, axis=1))

        qpts_qv = np.dot(qpts_qc, q0cell_cv)
        qpts_q = np.sum(qpts_qv**2, axis=1)**0.5
        qpts_q[qgamma] = 1e-14
        qdir_qv = qpts_qv / qpts_q[:, np.newaxis]
        qdir_qvv = qdir_qv[:, :, np.newaxis] * qdir_qv[:, np.newaxis, :]
        nq = len(qpts_qc)
        q0area = q0vol / q0cell_cv[2, 2]
        dq0 = q0area / nq
        dq0rad = (dq0 / pi)**0.5
        R = L / 2.
        x0area = q0area * R**2
        dx0rad = dq0rad * R

        exp_q = 4 * pi * (1 - np.exp(-qpts_q * R))
        dv_G = ((pi * L * G2_G * np.exp(-Gpar_G * R) * np.cos(G_Gv[:, 2] * R) -
                 4 * pi * Gpar_G * (1 - np.exp(-Gpar_G * R) *
                                    np.cos(G_Gv[:, 2] * R))) /
                (G2_G**1.5 * Gpar_G *
                 (4 * pi * (1 - np.exp(-Gpar_G * R) *
                            np.cos(G_Gv[:, 2] * R)))**0.5))

        dv_Gv = dv_G[:, np.newaxis] * G_Gv

        # Add corrections
        W_GG[:, 0] = 0.0
        W_GG[0, :] = 0.0

        A_q = np.sum(qdir_qv * np.dot(qdir_qv, L_vv), axis=1)
        frac_q = 1. / (1 + exp_q * A_q)

        # HEAD:
        w00_q = -(exp_q / qpts_q)**2 * A_q * frac_q
        w00_q[qgamma] = 0.0
        W_GG[0, 0] = w00_q.sum() / nq
        Axy = 0.5 * (L_vv[0, 0] + L_vv[1, 1])  # in-plane average
        a0 = 4 * pi * Axy + 1

        W_GG[0, 0] += -((a0 * dx0rad - np.log(a0 * dx0rad + 1)) /
                        a0**2 / x0area * 2 * np.pi * (2 * pi * L)**2 * Axy)

        # WINGS:
        u_q = -exp_q / qpts_q * frac_q
        W_GG[1:, 0] = 1. / nq * np.dot(
            np.sum(qdir_qv * u_q[:, np.newaxis], axis=0),
            S_vG0 * sqrV_G[np.newaxis, 1:])

        W_GG[0, 1:] = 1. / nq * np.dot(
            np.sum(qdir_qv * u_q[:, np.newaxis], axis=0),
            S_v0G * sqrV_G[np.newaxis, 1:])

        # BODY:
        # Constant corrections:
        W_GG[1:, 1:] += 1. / nq * sqrV_G[1:, None] * sqrV_G[None, 1:] * \
            np.tensordot(S_v0G, np.dot(S_vG0.T,
                                       np.sum(-qdir_qvv *
                                              exp_q[:, None, None] *
                                              frac_q[:, None, None],
                                              axis=0)), axes=(0, 1))
        u_vvv = np.tensordot(u_q[:, None] * qpts_qv, qdir_qvv, axes=(0, 0))
        # Gradient corrections:
        W_GG[1:, 1:] += 1. / nq * np.sum(
            dv_Gv[:, :, None] * np.tensordot(
                S_v0G, np.tensordot(u_vvv, S_vG0 * sqrV_G[None, 1:],
                                    axes=(2, 0)), axes=(0, 1)), axis=1)

    def update_energies(self, mixing):
        """Updates the energies of the calculator with the quasi-particle
        energies."""
        shifts_skn = np.zeros(self.shape)
        na, nb = self.bands
        i1 = len(self.qp_iskn) - 2
        i2 = i1 + 1
        if i1 < 0:
            i1 = 0

        for kpt in self.calc.wfs.kpt_u:
            s = kpt.s
            if kpt.k in self.kpts:
                ik = self.kpts.index(kpt.k)
                eps1_n = self.mixer(self.qp_iskn[i1, s, ik],
                                    self.qp_iskn[i2, s, ik],
                                    mixing)
                kpt.eps_n[na:nb] = eps1_n
                """
                Should we do something smart with the bands outside the
                interval?
                Here we shift the unoccupied bands not included by the average
                change of the top-most band and the occupied by the
                bottom-most band included
                """
                shifts_skn[s, ik] = (eps1_n - self.eps0_skn[s, kpt.k, na:nb])

        for kpt in self.calc.wfs.kpt_u:
            s = kpt.s
            if kpt.k in self.kpts:
                ik = self.kpts.index(kpt.k)
                kpt.eps_n[:na] = (self.eps0_skn[s, kpt.k, :na] +
                                  np.mean(shifts_skn[s, :, 0]))
                kpt.eps_n[nb:] = (self.eps0_skn[s, kpt.k, nb:] +
                                  np.mean(shifts_skn[s, :, -1]))
            else:
                """
                kpt.eps_n[:na] = (self.eps0_skn[s, kpt.k, :na] +
                                  np.mean(shifts_skn[s, :, 0]))
                kpt.eps_n[na:nb] = (self.eps0_skn[s, kpt.k, na:nb] +
                                    np.mean(shifts_skn[s, :], axis=0))
                kpt.eps_n[nb:] = (self.eps0_skn[s, kpt.k, nb:] +
                                  np.mean(shifts_skn[s, :, -1]))
                """
                pass

    def mixer(self, e0_skn, e1_skn, mixing=1.0):
        """Mix energies."""
        return e0_skn + mixing * (e1_skn - e0_skn)
