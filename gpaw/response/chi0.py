from __future__ import print_function, division
import numbers
from time import ctime

import numpy as np
from ase.units import Hartree
from ase.utils import devnull
from ase.utils.timing import timer, Timer

import gpaw.mpi as mpi
from gpaw import extra_parameters
from gpaw.blacs import (BlacsGrid, BlacsDescriptor, Redistributor,
                        DryRunBlacsGrid)
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.response.pair import PairDensity
from gpaw.utilities.memory import maxrss
from gpaw.utilities.blas import gemm
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.response.pair import PWSymmetryAnalyzer
from gpaw.response.integrators import (PointIntegrator, TetrahedronIntegrator)
from gpaw.bztools import convex_hull_volume

from functools import partial


class ArrayDescriptor:
    """Describes a single dimensional array."""

    def __init__(self, data_x):
        self.data_x = np.array(np.sort(data_x))
        self._data_len = len(data_x)

    def __len__(self):
        return self._data_len

    def get_data(self):
        return self.data_x

    def get_closest_index(self, scalars_w):
        """Get closest index.

        Get closest index approximating scalars from below."""
        diff_xw = self.data_x[:, np.newaxis] - scalars_w[np.newaxis]
        return np.argmin(diff_xw, axis=0)

    def get_index_range(self, lim1_m, lim2_m):
        """Get index range. """

        i0_m = np.zeros(len(lim1_m), int)
        i1_m = np.zeros(len(lim2_m), int)

        for m, (lim1, lim2) in enumerate(zip(lim1_m, lim2_m)):
            i_x = np.logical_and(lim1 <= self.data_x,
                                 lim2 >= self.data_x)
            if i_x.any():
                inds = np.argwhere(i_x)
                i0_m[m] = inds.min()
                i1_m[m] = inds.max() + 1

        return i0_m, i1_m


class FrequencyDescriptor(ArrayDescriptor):

    def __init__(self, domega0, omega2, omegamax):
        beta = (2**0.5 - 1) * domega0 / omega2
        wmax = int(omegamax / (domega0 + beta * omegamax))
        w = np.arange(wmax + 2)  # + 2 is for buffer
        omega_w = w * domega0 / (1 - beta * w)

        ArrayDescriptor.__init__(self, omega_w)

        self.domega0 = domega0
        self.omega2 = omega2
        self.omegamax = omegamax
        self.omegamin = 0

        self.beta = beta
        self.wmax = wmax
        self.omega_w = omega_w
        self.wmax = wmax
        self.nw = len(omega_w)

    def get_closest_index(self, o_m):
        beta = self.beta
        w_m = (o_m / (self.domega0 + beta * o_m)).astype(int)
        if isinstance(w_m, np.ndarray):
            w_m[w_m >= self.wmax] = self.wmax - 1
        elif isinstance(w_m, numbers.Integral):
            if w_m >= self.wmax:
                w_m = self.wmax - 1
        else:
            raise TypeError
        return w_m

    def get_index_range(self, omega1_m, omega2_m):
        omega1_m = omega1_m.copy()
        omega2_m = omega2_m.copy()
        omega1_m[omega1_m < 0] = 0
        omega2_m[omega2_m < 0] = 0
        w1_m = self.get_closest_index(omega1_m)
        w2_m = self.get_closest_index(omega2_m)
        o1_m = self.omega_w[w1_m]
        o2_m = self.omega_w[w2_m]
        w1_m[o1_m < omega1_m] += 1
        w2_m[o2_m < omega2_m] += 1
        return w1_m, w2_m


def frequency_grid(domega0, omega2, omegamax):
    beta = (2**0.5 - 1) * domega0 / omega2
    wmax = int(omegamax / (domega0 + beta * omegamax)) + 2
    w = np.arange(wmax)
    omega_w = w * domega0 / (1 - beta * w)
    return omega_w


class Chi0:
    """Class for calculating non-interacting response functions."""

    def __init__(self, calc, response='density',
                 frequencies=None, domega0=0.1, omega2=10.0, omegamax=None,
                 ecut=50, gammacentered=False, hilbert=True, nbands=None,
                 timeordered=False, eta=0.2, ftol=1e-6, threshold=1,
                 real_space_derivatives=False, intraband=True,
                 world=mpi.world, txt='-', timer=None,
                 nblocks=1, gate_voltage=None,
                 disable_point_group=False, disable_time_reversal=False,
                 disable_non_symmorphic=True,
                 scissor=None, integrationmode=None,
                 pbc=None, rate=0.0, eshift=0.0):
        """Construct Chi0 object.
        
        Parameters
        ----------
        calc : str
            The groundstate calculation file that the linear response
            calculation is based on.
        response : str
            Type of response function. Currently collinear, scalar options
            'density', '+-' and '-+' are implemented.
        frequencies : ndarray or None
            Array of frequencies to evaluate the response function at. If None,
            frequencies are determined using the frequency_grid function in
            gpaw.response.chi0.
        domega0, omega2, omegamax : float
            Input parameters for frequency_grid.
        ecut : float
            Energy cutoff.
        gammacentered : bool
            Center the grid of plane waves around the gamma point or q-vector
        hilbert : bool
            Switch for hilbert transform. If True, the full density response
            is determined from a hilbert transform of its spectral function.
            This is typically much faster, but does not work for imaginary
            frequencies.
        nbands : int
            Maximum band index to include.
        timeordered : bool
            Switch for calculating the time ordered density response function.
            In this case the hilbert transform cannot be used.
        eta : float
            Artificial broadening of spectra.
        ftol : float
            Threshold determining whether a band is completely filled
            (f > 1 - ftol) or completely empty (f < ftol).
        threshold : float
            Numerical threshold for the optical limit k dot p perturbation
            theory expansion (used in gpaw/response/pair.py).
        real_space_derivatives : bool
            Switch for calculating nabla matrix elements (in the optical limit)
            using a real space finite difference approximation.
        intraband : bool
            Switch for including the intraband contribution to the density
            response function.
        world : MPI comm instance
            MPI communicator.
        txt : str
            Output file.
        timer : gpaw.utilities.timing.timer instance
        nblocks : int
            Divide the response function into nblocks. Useful when the response
            function is large.
        gate_voltage : float
            Shift the fermi level by gate_voltage [Hartree].
        disable_point_group : bool
            Do not use the point group symmetry operators.
        disable_time_reversal : bool
            Do not use time reversal symmetry.
        disable_non_symmorphic : bool
            Do no use non symmorphic symmetry operators.
        scissor : tuple ([bands], shift [eV])
            Use scissor operator on bands.
        integrationmode : str
            Integrator for the kpoint integration.
            If == 'tetrahedron integration' then the kpoint integral is
            performed using the linear tetrahedron method.
        pbc : list
            Periodic directions of the system. Defaults to [True, True, True].
        eshift : float
            Shift unoccupied bands

        Attributes
        ----------
        pair : gpaw.response.pair.PairDensity instance
            Class for calculating matrix elements of pairs of wavefunctions.

        """
        
        self.response = response
        
        self.timer = timer or Timer()

        self.pair = PairDensity(calc, ecut, self.response,
                                ftol, threshold,
                                real_space_derivatives, world, txt,
                                self.timer,
                                nblocks=nblocks, gate_voltage=gate_voltage)

        self.disable_point_group = disable_point_group
        self.disable_time_reversal = disable_time_reversal
        self.disable_non_symmorphic = disable_non_symmorphic
        self.integrationmode = integrationmode
        self.eshift = eshift / Hartree

        calc = self.pair.calc
        self.calc = calc

        if world.rank != 0:
            txt = devnull
        elif isinstance(txt, str):
            txt = open(txt, 'w')
        self.fd = txt

        self.vol = abs(np.linalg.det(calc.wfs.gd.cell_cv))

        self.world = world

        if nblocks == 1:
            self.blockcomm = self.world.new_communicator([world.rank])
            self.kncomm = world
        else:
            assert world.size % nblocks == 0, world.size
            rank1 = world.rank // nblocks * nblocks
            rank2 = rank1 + nblocks
            self.blockcomm = self.world.new_communicator(range(rank1, rank2))
            ranks = range(world.rank % nblocks, world.size, nblocks)
            self.kncomm = self.world.new_communicator(ranks)

        self.nblocks = nblocks

        if world.rank != 0:
            txt = devnull
        elif isinstance(txt, str):
            txt = open(txt, 'w')
        self.fd = txt

        if ecut is not None:
            ecut /= Hartree

        self.ecut = ecut
        self.gammacentered = gammacentered
        
        self.eta = eta / Hartree
        if rate == 'eta':
            self.rate = self.eta
        else:
            self.rate = rate / Hartree
        self.domega0 = domega0 / Hartree
        self.omega2 = omega2 / Hartree
        self.omegamax = None if omegamax is None else omegamax / Hartree
        self.nbands = nbands or self.calc.wfs.bd.nbands
        self.include_intraband = intraband

        omax = self.find_maximum_frequency()
        
        if frequencies is None:
            if self.omegamax is None:
                self.omegamax = omax
            print('Using nonlinear frequency grid from 0 to %.3f eV' %
                  (self.omegamax * Hartree), file=self.fd)
            self.wd = FrequencyDescriptor(self.domega0, self.omega2,
                                          self.omegamax)
        else:
            self.wd = ArrayDescriptor(np.asarray(frequencies) / Hartree)
            assert not hilbert

        self.omega_w = self.wd.get_data()
        self.hilbert = hilbert
        self.timeordered = bool(timeordered)

        if self.eta == 0.0:
            assert not hilbert
            assert not timeordered
            assert not self.omega_w.real.any()

        self.nocc1 = self.pair.nocc1  # number of completely filled bands
        self.nocc2 = self.pair.nocc2  # number of non-empty bands

        self.Q_aGii = None

        if pbc is not None:
            self.pbc = np.array(pbc)
        else:
            self.pbc = np.array([True, True, True])

        if self.pbc is not None and (~self.pbc).any():
            assert np.sum((~self.pbc).astype(int)) == 1, \
                print('Only one non-periodic direction supported atm.')
            print('Nonperiodic BC\'s: ', (~self.pbc),
                  file=self.fd)

        if integrationmode is not None:
            print('Using integration method: ' + self.integrationmode,
                  file=self.fd)
        else:
            print('Using integration method: PointIntegrator', file=self.fd)
    
    def find_maximum_frequency(self):
        """Determine the maximum electron-hole pair transition energy."""
        self.epsmin = 10000.0
        self.epsmax = -10000.0
        for kpt in self.calc.wfs.kpt_u:
            self.epsmin = min(self.epsmin, kpt.eps_n[0])
            self.epsmax = max(self.epsmax, kpt.eps_n[self.nbands - 1])

        print('Minimum eigenvalue: %10.3f eV' % (self.epsmin * Hartree),
              file=self.fd)
        print('Maximum eigenvalue: %10.3f eV' % (self.epsmax * Hartree),
              file=self.fd)

        return self.epsmax - self.epsmin

    def calculate(self, q_c, spin='all', A_x=None):
        """Calculate response function.

        Parameters
        ----------
        q_c : list or ndarray
            Momentum vector.
        spin : str or int
            If 'all' then include all spins.
            If 0 or 1, only include this specific spin.
            (not used in transverse reponse functions)
        A_x : ndarray
            Output array. If None, the output array is created.

        Returns
        -------
        pd : Planewave descriptor
            Planewave descriptor for q_c.
        chi0_wGG : ndarray
            The response function.
        chi0_wxvG : ndarray or None
            (Only in optical limit) Wings of the density response function.
        chi0_wvv : ndarray or None
            (Only in optical limit) Head of the density response function.

        """
        wfs = self.calc.wfs

        if self.response == 'density':
            if spin == 'all':
                spins = range(wfs.nspins)
            else:
                assert spin in range(wfs.nspins)
                spins = [spin]
        else:
            if self.response == '+-':
                spins = [0]
            elif self.response == '-+':
                spins = [1]
            else:
                raise ValueError('Invalid response %s' % self.response)

        q_c = np.asarray(q_c, dtype=float)
        optical_limit = np.allclose(q_c, 0.0) and self.response == 'density'

        pd = self.get_PWDescriptor(q_c, self.gammacentered)

        self.print_chi(pd)

        if extra_parameters.get('df_dry_run'):
            print('    Dry run exit', file=self.fd)
            raise SystemExit

        nG = pd.ngmax + 2 * optical_limit
        nw = len(self.omega_w)
        mynG = (nG + self.blockcomm.size - 1) // self.blockcomm.size
        self.Ga = min(self.blockcomm.rank * mynG, nG)
        self.Gb = min(self.Ga + mynG, nG)
        # if self.blockcomm.rank == 0:
        #     assert self.Gb - self.Ga >= 3
        # assert mynG * (self.blockcomm.size - 1) < nG
        if A_x is not None:
            nx = nw * (self.Gb - self.Ga) * nG
            chi0_wGG = A_x[:nx].reshape((nw, self.Gb - self.Ga, nG))
            chi0_wGG[:] = 0.0
        else:
            chi0_wGG = np.zeros((nw, self.Gb - self.Ga, nG), complex)
            
        if optical_limit:
            chi0_wxvG = np.zeros((len(self.omega_w), 2, 3, nG), complex)
            chi0_wvv = np.zeros((len(self.omega_w), 3, 3), complex)
            self.plasmafreq_vv = np.zeros((3, 3), complex)
        else:
            chi0_wxvG = None
            chi0_wvv = None
            self.plasmafreq_vv = None

        if self.response == 'density':
            # Do all empty bands:
            m1 = self.nocc1
        else:
            # Do all bands
            m1 = 0
        
        m2 = self.nbands

        pd, chi0_wGG, chi0_wxvG, chi0_wvv = self._calculate(pd,
                                                            chi0_wGG,
                                                            chi0_wxvG,
                                                            chi0_wvv,
                                                            m1, m2, spins)
        
        return pd, chi0_wGG, chi0_wxvG, chi0_wvv

    @timer('Calculate CHI_0')
    def _calculate(self, pd, chi0_wGG, chi0_wxvG, chi0_wvv, m1, m2, spins,
                   extend_head=True):
        """In-place calculation of the response function.
        
        Parameters
        ----------
        q_c : list or ndarray
            Momentum vector..
        pd : Planewave descriptor
            Planewave descriptor for q_c.
        chi0_wGG : ndarray
            The response function.
        chi0_wxvG : ndarray or None
            Wings of the density response function.
        chi0_wvv : ndarray or None
            Head of the density response function.
        m1 : int
            Lower band cutoff for band summation
        m2 : int
            Upper band cutoff for band summation
        spins : str or list(ints)
            If 'all' then include all spins.
            If [0] or [1], only include this specific spin (and flip it,
            if calculating the transverse magnetic response).
        extend_head : bool
            If True: Extend the wings and head of chi in the optical limit to
            take into account the non-analytic nature of chi. Effectively
            means that chi has dimension (nw, nG + 2, nG + 2) in the optical
            limit. This simplifies the code and should only be switched off
            for parts of the code that do not support this feature i.e., GW
            RPA total energy and RALDA.
        """
        
        # Parse spins
        wfs = self.calc.wfs
        if spins == 'all':
            spins = range(wfs.nspins)
        elif spins == 'pm':
            spins = [0]
        elif spins == 'mp':
            spins = [1]
        else:
            for spin in spins:
                assert spin in range(wfs.nspins)

        # Are we calculating the optical limit.
        optical_limit = np.allclose(pd.kd.bzk_kc[0], 0.0) and \
            self.response == 'density'

        # Reset PAW correction in case momentum has change
        self.Q_aGii = self.pair.initialize_paw_corrections(pd)
        A_wxx = chi0_wGG  # Change notation

        # Initialize integrator. The integrator class is a general class
        # for brillouin zone integration that can integrate user defined
        # functions over user defined domains and sum over bands.
        if self.integrationmode is None or \
           self.integrationmode == 'point integration':
            integrator = PointIntegrator(self.pair.calc.wfs.gd.cell_cv,
                                         response=self.response,
                                         comm=self.world,
                                         timer=self.timer,
                                         txt=self.fd,
                                         eshift=self.eshift,
                                         nblocks=self.nblocks)
            intnoblock = PointIntegrator(self.pair.calc.wfs.gd.cell_cv,
                                         response=self.response,
                                         comm=self.world,
                                         timer=self.timer,
                                         eshift=self.eshift,
                                         txt=self.fd)
        elif self.integrationmode == 'tetrahedron integration':
            integrator = TetrahedronIntegrator(self.pair.calc.wfs.gd.cell_cv,
                                               response=self.response,
                                               comm=self.world,
                                               timer=self.timer,
                                               eshift=self.eshift,
                                               txt=self.fd,
                                               nblocks=self.nblocks)
            intnoblock = TetrahedronIntegrator(self.pair.calc.wfs.gd.cell_cv,
                                               response=self.response,
                                               comm=self.world,
                                               timer=self.timer,
                                               eshift=self.eshift,
                                               txt=self.fd)
        else:
            print('Integration mode ' + self.integrationmode +
                  ' not implemented.', file=self.fd)
            raise NotImplementedError

        # The integration domain is determined by the following function
        # that reduces the integration domain to the irreducible zone
        # of the little group of q.
        bzk_kv, PWSA = self.get_kpoints(pd,
                                        integrationmode=self.integrationmode)
        domain = (bzk_kv, spins)

        if self.integrationmode == 'tetrahedron integration':
            # If there are non-periodic directions it is possible that the
            # integration domain is not compatible with the symmetry operations
            # which essentially means that too large domains will be
            # integrated. We normalize by vol(BZ) / vol(domain) to make
            # sure that to fix this.
            domainvol = convex_hull_volume(bzk_kv) * PWSA.how_many_symmetries()
            bzvol = (2 * np.pi)**3 / self.vol
            factor = bzvol / domainvol
        else:
            factor = 1

        prefactor = (2 * factor * PWSA.how_many_symmetries() /
                     (wfs.nspins * (2 * np.pi)**3))  # Remember prefactor
        
        if self.integrationmode is None:
            if self.calc.wfs.kd.refine_info is not None:
                nbzkpts = self.calc.wfs.kd.refine_info.mhnbzkpts
            else:
                nbzkpts = self.calc.wfs.kd.nbzkpts
            prefactor *= len(bzk_kv) / nbzkpts

        # The functions that are integrated are defined in the bottom
        # of this file and take a number of constant keyword arguments
        # which the integrator class accepts through the use of the
        # kwargs keyword.
        kd = self.calc.wfs.kd
        mat_kwargs = {'kd': kd, 'pd': pd, 'n1': 0,
                      'm1': m1, 'm2': m2,
                      'symmetry': PWSA,
                      'integrationmode': self.integrationmode}
        eig_kwargs = {'kd': kd, 'm1': m1, 'm2': m2,
                      'n1': 0, 'pd': pd}
        if self.response == 'density':
            mat_kwargs['n2'] = self.nocc2
            eig_kwargs['n2'] = self.nocc2
        else:
            mat_kwargs['n2'] = self.nbands
            eig_kwargs['n2'] = self.nbands
        
        if not extend_head:
            mat_kwargs['extend_head'] = False

        # Determine what "kind" of integral to make.
        extraargs = {}  # Initialize extra arguments to integration method.
        if self.eta == 0:
            # If eta is 0 then we must be working with imaginary frequencies.
            # In this case chi is hermitian and it is therefore possible to
            # reduce the computational costs by a only computing half of the
            # response function.
            kind = 'hermitian response function'
        elif self.hilbert:
            # The spectral function integrator assumes that the form of the
            # integrand is a function (a matrix element) multiplied by
            # a delta function and should return a function of at user defined
            # x's (frequencies). Thus the integrand is tuple of two functions
            # and takes an additional argument (x).
            kind = 'spectral function'
        else:
            # Otherwise, we can make no simplifying assumptions of the
            # form of the response function and we simply perform a brute
            # force calculation of the response function.
            kind = 'response function'
            extraargs['eta'] = self.eta
            extraargs['timeordered'] = self.timeordered

        if optical_limit and not extend_head:
            wings = True
        else:
            wings = False

        A_wxx /= prefactor
        if wings:
            chi0_wxvG /= prefactor
            chi0_wvv /= prefactor
        
        # Integrate response function
        print('Integrating response function.', file=self.fd)
        integrator.integrate(kind=kind,  # Kind of integral
                             domain=domain,  # Integration domain
                             integrand=(self.get_matrix_element,  # Integrand
                                        self.get_eigenvalues),  # Integrand
                             x=self.wd,  # Frequency Descriptor
                             kwargs=(mat_kwargs, eig_kwargs),
                             # Arguments for integrand functions
                             out_wxx=A_wxx,  # Output array
                             **extraargs)
        # extraargs: Extra arguments to integration method
        if wings:
            mat_kwargs['extend_head'] = True
            mat_kwargs['block'] = False
            if self.eta == 0:
                extraargs['eta'] = self.eta
            # This is horrible but we need to update the wings manually
            # in order to make them work with ralda, RPA and GW. This entire
            # section can be deleted in the future if the ralda and RPA code is
            # made compatible with the head and wing extension that other parts
            # of the code is using.
            chi0_wxvx = np.zeros(np.array(chi0_wxvG.shape) +
                                 [0, 0, 0, 2],
                                 complex)  # Notice the wxv"x" for head extend
            intnoblock.integrate(kind=kind + ' wings',  # kind'o int.
                                 domain=domain,  # Integration domain
                                 integrand=(self.get_matrix_element,  # Intgrnd
                                            self.get_eigenvalues),  # Integrand
                                 x=self.wd,  # Frequency Descriptor
                                 kwargs=(mat_kwargs, eig_kwargs),
                                 # Arguments for integrand functions
                                 out_wxx=chi0_wxvx,  # Output array
                                 **extraargs)

        if self.hilbert:
            # The integrator only returns the spectral function and a Hilbert
            # transform is performed to return the real part of the density
            # response function.
            with self.timer('Hilbert transform'):
                omega_w = self.wd.get_data()  # Get frequencies
                # Make Hilbert transform
                ht = HilbertTransform(np.array(omega_w), self.eta,
                                      timeordered=self.timeordered)
                ht(A_wxx)
                if wings:
                    ht(chi0_wxvx)

        # In the optical limit additional work must be performed
        # for the intraband response.
        # Only compute the intraband response if there are partially
        # unoccupied bands and only if the user has not disabled its
        # calculation using the include_intraband keyword.
        if optical_limit and self.nocc1 != self.nocc2:
            # The intraband response is essentially just the calculation
            # of the free space Drude plasma frequency. The calculation is
            # similarly to the interband transitions documented above.
            mat_kwargs = {'kd': kd, 'symmetry': PWSA,
                          'n1': self.nocc1, 'n2': self.nocc2,
                          'pd': pd}  # Integrand arguments
            eig_kwargs = {'kd': kd,
                          'n1': self.nocc1, 'n2': self.nocc2,
                          'pd': pd}  # Integrand arguments
            domain = (bzk_kv, spins)  # Integration domain
            fermi_level = self.pair.fermi_level  # Fermi level

            # Not so elegant solution but it works
            plasmafreq_wvv = np.zeros((1, 3, 3), complex)  # Output array
            print('Integrating intraband density response.', file=self.fd)

            # Depending on which integration method is used we
            # have to pass different arguments
            extraargs = {}
            if self.integrationmode is None:
                # Calculate intraband transitions at finite fermi smearing
                extraargs['intraband'] = True  # Calculate intraband
            elif self.integrationmode == 'tetrahedron integration':
                # Calculate intraband transitions at T=0
                extraargs['x'] = ArrayDescriptor([-fermi_level])

            intnoblock.integrate(kind='spectral function',  # Kind of integral
                                 domain=domain,  # Integration domain
                                 # Integrands
                                 integrand=(self.get_intraband_response,
                                            self.get_intraband_eigenvalue),
                                 # Integrand arguments
                                 kwargs=(mat_kwargs, eig_kwargs),
                                 out_wxx=plasmafreq_wvv,  # Output array
                                 **extraargs)  # Extra args for int. method

            # Again, not so pretty but that's how it is
            plasmafreq_vv = plasmafreq_wvv[0].copy()
            if self.include_intraband:
                if extend_head:
                    va = min(self.Ga, 3)
                    vb = min(self.Gb, 3)
                    A_wxx[:, :vb - va, :3] += (plasmafreq_vv[va:vb] /
                                               (self.omega_w[:, np.newaxis,
                                                             np.newaxis] +
                                                1e-10 + self.rate * 1j)**2)
                elif self.blockcomm.rank == 0:
                    A_wxx[:, 0, 0] += (plasmafreq_vv[2, 2] /
                                       (self.omega_w + 1e-10 +
                                        self.rate * 1j)**2)

            # Save the plasmafrequency
            try:
                self.plasmafreq_vv += 4 * np.pi * plasmafreq_vv * prefactor
            except AttributeError:
                self.plasmafreq_vv = 4 * np.pi * plasmafreq_vv * prefactor

            PWSA.symmetrize_wvv(self.plasmafreq_vv[np.newaxis])
            print('Plasma frequency:', file=self.fd)
            print((self.plasmafreq_vv**0.5 * Hartree).round(2),
                  file=self.fd)

        # The response function is integrated only over the IBZ. The
        # chi calculated above must therefore be extended to include the
        # response from the full BZ. This extension can be performed as a
        # simple post processing of the response function that makes
        # sure that the response function fulfills the symmetries of the little
        # group of q. Due to the specific details of the implementation the chi
        # calculated above is normalized by the number of symmetries (as seen
        # below) and then symmetrized.
        A_wxx *= prefactor
        
        tmpA_wxx = self.redistribute(A_wxx)
        if extend_head:
            PWSA.symmetrize_wxx(tmpA_wxx,
                                optical_limit=optical_limit)
        else:
            PWSA.symmetrize_wGG(tmpA_wxx)
            if wings:
                chi0_wxvG += chi0_wxvx[..., 2:]
                chi0_wvv += chi0_wxvx[:, 0, :3, :3]
                PWSA.symmetrize_wxvG(chi0_wxvG)
                PWSA.symmetrize_wvv(chi0_wvv)
        self.redistribute(tmpA_wxx, A_wxx)

        # If point summation was used then the normalization of the
        # response function is not right and we have to make up for this
        # fact.

        if wings:
            chi0_wxvG *= prefactor
            chi0_wvv *= prefactor

        # In the optical limit, we have extended the wings and the head to
        # account for their nonanalytic behaviour which means that the size of
        # the chi0_wGG matrix is nw * (nG + 2)**2. Below we extract these
        # parameters.

        if optical_limit and extend_head:
            # The wings are extracted
            chi0_wxvG[:, 1, :, self.Ga:self.Gb] = np.transpose(A_wxx[..., 0:3],
                                                               (0, 2, 1))
            va = min(self.Ga, 3)
            vb = min(self.Gb, 3)
            # print(self.world.rank, va, vb, chi0_wxvG[:, 0, va:vb].shape,
            #       A_wxx[:, va:vb].shape, A_wxx.shape)
            chi0_wxvG[:, 0, va:vb] = A_wxx[:, :vb - va]

            # Add contributions on different ranks
            self.blockcomm.sum(chi0_wxvG)
            chi0_wvv[:] = chi0_wxvG[:, 0, :3, :3]
            chi0_wxvG = chi0_wxvG[..., 2:]  # Jesus, this is complicated

            # The head is extracted
            # if self.blockcomm.rank == 0:
            #     chi0_wvv[:] = A_wxx[:, :3, :3]
            # self.blockcomm.broadcast(chi0_wvv, 0)

            # It is easiest to redistribute over freqs to pick body
            tmpA_wxx = self.redistribute(A_wxx)
            chi0_wGG = tmpA_wxx[:, 2:, 2:]
            chi0_wGG = self.redistribute(chi0_wGG)

        elif optical_limit:
            # Since chi_wGG is nonanalytic in the head
            # and wings we have to take care that
            # these are handled correctly. Note that
            # it is important that the wings are overwritten first.
            chi0_wGG[:, :, 0] = chi0_wxvG[:, 1, 2, self.Ga:self.Gb]

            if self.blockcomm.rank == 0:
                chi0_wGG[:, 0, :] = chi0_wxvG[:, 0, 2, :]
                chi0_wGG[:, 0, 0] = chi0_wvv[:, 2, 2]
        else:
            chi0_wGG = A_wxx

        return pd, chi0_wGG, chi0_wxvG, chi0_wvv

    def get_PWDescriptor(self, q_c, gammacentered=False):
        """Get the planewave descriptor of q_c."""
        qd = KPointDescriptor([q_c])
        pd = PWDescriptor(self.ecut, self.calc.wfs.gd,
                          complex, qd, gammacentered=gammacentered)
        return pd

    @timer('Get kpoints')
    def get_kpoints(self, pd, integrationmode=None):
        """Get the integration domain."""
        # Use symmetries
        PWSA = PWSymmetryAnalyzer
        PWSA = PWSA(self.calc.wfs.kd, pd,
                    timer=self.timer, txt=self.fd,
                    disable_point_group=self.disable_point_group,
                    disable_time_reversal=self.disable_time_reversal,
                    disable_non_symmorphic=self.disable_non_symmorphic)

        if integrationmode is None:
            K_gK = PWSA.group_kpoints()
            bzk_kc = np.array([self.calc.wfs.kd.bzk_kc[K_K[0]] for
                               K_K in K_gK])
        elif integrationmode == 'tetrahedron integration':
            bzk_kc = PWSA.get_reduced_kd(pbc_c=self.pbc).bzk_kc
            if (~self.pbc).any():
                bzk_kc = np.append(bzk_kc,
                                   bzk_kc + (~self.pbc).astype(int),
                                   axis=0)

        bzk_kv = np.dot(bzk_kc, pd.gd.icell_cv) * 2 * np.pi

        return bzk_kv, PWSA

    @timer('Get matrix element')
    def get_matrix_element(self, k_v, s, n1=None, n2=None,
                           m1=None, m2=None,
                           pd=None, kd=None,
                           symmetry=None, integrationmode=None,
                           extend_head=True, block=True):
        """A function that returns pair-densities.

        A pair density is defined as::

         <snk| e^(-i (q + G) r) |s'mk+q>,

        where s and s' are spins, n and m are band indices, k is
        the kpoint and q is the momentum transfer. For dielectric
        reponse s'=s, for the transverse magnetic reponse
        s' is flipped with respect to s.
        
        Parameters
        ----------
        k_v : ndarray
            Kpoint coordinate in cartesian coordinates.
        s : int
            Spin index.
        n1 : int
            Lower occupied band index.
        n2 : int
            Upper occupied band index.
        m1 : int
            Lower unoccupied band index.
        m2 : int
            Upper unoccupied band index.
        pd : PlanewaveDescriptor instance
        kd : KpointDescriptor instance
            Calculator kpoint descriptor.
        symmetry: gpaw.response.pair.PWSymmetryAnalyzer instance
            PWSA object for handling symmetries of the kpoints.
        integrationmode : str
            The integration mode employed.
        extend_head: Bool
            Extend the head to include non-analytic behaviour

        Return
        ------
        n_nmG : ndarray
            Pair densities.
        """
        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)

        q_c = pd.kd.bzk_kc[0]

        optical_limit = np.allclose(q_c, 0.0) and self.response == 'density'

        extrapolate_q = False
        if self.calc.wfs.kd.refine_info is not None:
            K1 = self.pair.find_kpoint(k_c)
            label = kd.refine_info.label_k[K1]
            if label == 'zero':
                return None
            elif (kd.refine_info.almostoptical and label == 'mh'):
                if not hasattr(self, 'pd0'):
                    self.pd0 = self.get_PWDescriptor([0, ] * 3)
                pd = self.pd0
                extrapolate_q = True

        nG = pd.ngmax
        weight = np.sqrt(symmetry.get_kpoint_weight(k_c) /
                         symmetry.how_many_symmetries())
        if self.Q_aGii is None:
            self.Q_aGii = self.pair.initialize_paw_corrections(pd)

        kptpair = self.pair.get_kpoint_pair(pd, s, k_c, n1, n2,
                                            m1, m2, block=block)

        m_m = np.arange(m1, m2)
        n_n = np.arange(n1, n2)

        n_nmG = self.pair.get_pair_density(pd, kptpair, n_n, m_m,
                                           Q_aGii=self.Q_aGii, block=block)
        
        if integrationmode is None:
            n_nmG *= weight
        
        df_nm = kptpair.get_occupation_differences(n_n, m_m)
        if not self.response == 'density':
            df_nm = np.abs(df_nm)
        df_nm[df_nm <= 1e-20] = 0.0
        n_nmG *= df_nm[..., np.newaxis]**0.5

        if extrapolate_q:
            q_v = np.dot(q_c, pd.gd.icell_cv) * (2 * np.pi)
            nq_nm = np.dot(n_nmG[:, :, :3], q_v)
            n_nmG = n_nmG[:, :, 2:]
            n_nmG[:, :, 0] = nq_nm
        
        if not extend_head and optical_limit:
            n_nmG = np.copy(n_nmG[:, :, 2:])
            optical_limit = False

        if extend_head and optical_limit:
            return n_nmG.reshape(-1, nG + 2 * optical_limit)
        else:
            return n_nmG.reshape(-1, nG)

    @timer('Get eigenvalues')
    def get_eigenvalues(self, k_v, s, n1=None, n2=None,
                        m1=None, m2=None,
                        kd=None, pd=None, wfs=None,
                        filter=False):
        """A function that can return the eigenvalues.

        A simple function describing the integrand of
        the response function which gives an output that
        is compatible with the gpaw k-point integration
        routines."""
        if wfs is None:
            wfs = self.calc.wfs

        kd = wfs.kd
        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)
        q_c = pd.kd.bzk_kc[0]
        K1 = self.pair.find_kpoint(k_c)
        K2 = self.pair.find_kpoint(k_c + q_c)

        ik1 = kd.bz2ibz_k[K1]
        ik2 = kd.bz2ibz_k[K2]
        kpt1 = wfs.kpt_u[s * wfs.kd.nibzkpts + ik1]
        if self.response in ['+-', '-+']:
            s2 = 1 - s
        else:
            s2 = s
        kpt2 = wfs.kpt_u[s2 * wfs.kd.nibzkpts + ik2]
        deps_nm = np.subtract(kpt1.eps_n[n1:n2][:, np.newaxis],
                              kpt2.eps_n[m1:m2])

        if filter:
            fermi_level = self.pair.fermi_level
            deps_nm[kpt1.eps_n[n1:n2] > fermi_level, :] = np.nan
            deps_nm[:, kpt2.eps_n[m1:m2] < fermi_level] = np.nan

        return deps_nm.reshape(-1)

    def get_intraband_response(self, k_v, s, n1=None, n2=None,
                               kd=None, symmetry=None, pd=None,
                               integrationmode=None):
        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)
        kpt1 = self.pair.get_k_point(s, k_c, n1, n2)
        n_n = range(n1, n2)

        vel_nv = self.pair.intraband_pair_density(kpt1, n_n)

        if self.integrationmode is None:
            f_n = kpt1.f_n
            width = self.calc.occupations.width
            if width > 1e-15:
                dfde_n = - 1. / width * (f_n - f_n**2.0)
            else:
                dfde_n = np.zeros_like(f_n)
            vel_nv *= np.sqrt(-dfde_n[:, np.newaxis])
            weight = np.sqrt(symmetry.get_kpoint_weight(k_c) /
                             symmetry.how_many_symmetries())
            vel_nv *= weight

        return vel_nv

    @timer('Intraband eigenvalue')
    def get_intraband_eigenvalue(self, k_v, s,
                                 n1=None, n2=None, kd=None, pd=None):
        """A function that can return the eigenvalues.

        A simple function describing the integrand of
        the response function which gives an output that
        is compatible with the gpaw k-point integration
        routines."""
        wfs = self.calc.wfs
        kd = wfs.kd
        k_c = np.dot(pd.gd.cell_cv, k_v) / (2 * np.pi)
        K1 = self.pair.find_kpoint(k_c)
        ik = kd.bz2ibz_k[K1]
        kpt1 = wfs.kpt_u[s * wfs.kd.nibzkpts + ik]

        return kpt1.eps_n[n1:n2]

    @timer('redist')
    def redistribute(self, in_wGG, out_x=None):
        """Redistribute array.

        Switch between two kinds of parallel distributions:

        1) parallel over G-vectors (second dimension of in_wGG)
        2) parallel over frequency (first dimension of in_wGG)

        Returns new array using the memory in the 1-d array out_x.
        """

        comm = self.blockcomm

        if comm.size == 1:
            return in_wGG

        nw = len(self.omega_w)
        nG = in_wGG.shape[2]
        mynw = (nw + comm.size - 1) // comm.size
        mynG = (nG + comm.size - 1) // comm.size

        bg1 = BlacsGrid(comm, comm.size, 1)
        bg2 = BlacsGrid(comm, 1, comm.size)
        md1 = BlacsDescriptor(bg1, nw, nG**2, mynw, nG**2)
        md2 = BlacsDescriptor(bg2, nw, nG**2, nw, mynG * nG)

        if len(in_wGG) == nw:
            mdin = md2
            mdout = md1
        else:
            mdin = md1
            mdout = md2

        r = Redistributor(comm, mdin, mdout)

        outshape = (mdout.shape[0], mdout.shape[1] // nG, nG)
        if out_x is None:
            out_wGG = np.empty(outshape, complex)
        else:
            out_wGG = out_x[:np.product(outshape)].reshape(outshape)

        r.redistribute(in_wGG.reshape(mdin.shape),
                       out_wGG.reshape(mdout.shape))

        return out_wGG

    @timer('dist freq')
    def distribute_frequencies(self, chi0_wGG):
        """Distribute frequencies to all cores."""

        world = self.world
        comm = self.blockcomm

        if world.size == 1:
            return chi0_wGG

        nw = len(self.omega_w)
        nG = chi0_wGG.shape[2]
        mynw = (nw + world.size - 1) // world.size
        mynG = (nG + comm.size - 1) // comm.size

        wa = min(world.rank * mynw, nw)
        wb = min(wa + mynw, nw)

        if self.blockcomm.size == 1:
            return chi0_wGG[wa:wb].copy()

        if self.kncomm.rank == 0:
            bg1 = BlacsGrid(comm, 1, comm.size)
            in_wGG = chi0_wGG.reshape((nw, -1))
        else:
            bg1 = DryRunBlacsGrid(mpi.serial_comm, 1, 1)
            in_wGG = np.zeros((0, 0), complex)
        md1 = BlacsDescriptor(bg1, nw, nG**2, nw, mynG * nG)

        bg2 = BlacsGrid(world, world.size, 1)
        md2 = BlacsDescriptor(bg2, nw, nG**2, mynw, nG**2)

        r = Redistributor(world, md1, md2)
        shape = (wb - wa, nG, nG)
        out_wGG = np.empty(shape, complex)
        r.redistribute(in_wGG, out_wGG.reshape((wb - wa, nG**2)))

        return out_wGG
    
    def print_chi(self, pd):
        calc = self.calc
        gd = calc.wfs.gd

        if extra_parameters.get('df_dry_run'):
            from gpaw.mpi import DryRunCommunicator
            size = extra_parameters['df_dry_run']
            world = DryRunCommunicator(size)
        else:
            world = self.world

        q_c = pd.kd.bzk_kc[0]
        nw = len(self.omega_w)
        ecut = self.ecut * Hartree
        ns = calc.wfs.nspins
        nbands = self.nbands
        nk = calc.wfs.kd.nbzkpts
        nik = calc.wfs.kd.nibzkpts
        ngmax = pd.ngmax
        eta = self.eta * Hartree
        wsize = world.size
        knsize = self.kncomm.size
        nocc = self.nocc1
        npocc = self.nocc2
        ngridpoints = gd.N_c[0] * gd.N_c[1] * gd.N_c[2]
        nstat = (ns * npocc + world.size - 1) // world.size
        occsize = nstat * ngridpoints * 16. / 1024**2
        bsize = self.blockcomm.size
        chisize = nw * pd.ngmax**2 * 16. / 1024**2 / bsize

        p = partial(print, file=self.fd)

        p('%s' % ctime())
        p('Called response.chi0.calculate with')
        p('    q_c: [%f, %f, %f]' % (q_c[0], q_c[1], q_c[2]))
        p('    Number of frequency points: %d' % nw)
        p('    Planewave cutoff: %f' % ecut)
        p('    Number of spins: %d' % ns)
        p('    Number of bands: %d' % nbands)
        p('    Number of kpoints: %d' % nk)
        p('    Number of irredicible kpoints: %d' % nik)
        p('    Number of planewaves: %d' % ngmax)
        p('    Broadening (eta): %f' % eta)
        p('    world.size: %d' % wsize)
        p('    kncomm.size: %d' % knsize)
        p('    blockcomm.size: %d' % bsize)
        p('    Number of completely occupied states: %d' % nocc)
        p('    Number of partially occupied states: %d' % npocc)
        p()
        p('    Memory estimate of potentially large arrays:')
        p('        chi0_wGG: %f M / cpu' % chisize)
        p('        Occupied states: %f M / cpu' % occsize)
        p('        Memory usage before allocation: %f M / cpu' % (maxrss() /
                                                                  1024**2))
        p()


class HilbertTransform:

    def __init__(self, omega_w, eta, timeordered=False, gw=False,
                 blocksize=500):
        """Analytic Hilbert transformation using linear interpolation.

        Hilbert transform::

           oo
          /           1                1
          |dw' (-------------- - --------------) S(w').
          /     w - w' + i eta   w + w' + i eta
          0

        With timeordered=True, you get::

           oo
          /           1                1
          |dw' (-------------- - --------------) S(w').
          /     w - w' - i eta   w + w' + i eta
          0

        With gw=True, you get::

           oo
          /           1                1
          |dw' (-------------- + --------------) S(w').
          /     w - w' + i eta   w + w' + i eta
          0

        """

        self.blocksize = blocksize

        if timeordered:
            self.H_ww = self.H(omega_w, -eta) + self.H(omega_w, -eta, -1)
        elif gw:
            self.H_ww = self.H(omega_w, eta) - self.H(omega_w, -eta, -1)
        else:
            self.H_ww = self.H(omega_w, eta) + self.H(omega_w, -eta, -1)

    def H(self, o_w, eta, sign=1):
        """Calculate transformation matrix.

        With s=sign (+1 or -1)::

                        oo
                       /       dw'
          X (w, eta) = | ---------------- S(w').
           s           / s w - w' + i eta
                       0

        Returns H_ij so that X_i = np.dot(H_ij, S_j), where::

            X_i = X (omega_w[i]) and S_j = S(omega_w[j])
                   s
        """

        nw = len(o_w)
        H_ij = np.zeros((nw, nw), complex)
        do_j = o_w[1:] - o_w[:-1]
        for i, o in enumerate(o_w):
            d_j = o_w - o * sign
            y_j = 1j * np.arctan(d_j / eta) + 0.5 * np.log(d_j**2 + eta**2)
            y_j = (y_j[1:] - y_j[:-1]) / do_j
            H_ij[i, :-1] = 1 - (d_j[1:] - 1j * eta) * y_j
            H_ij[i, 1:] -= 1 - (d_j[:-1] - 1j * eta) * y_j
        return H_ij

    def __call__(self, S_wx):
        """Inplace transform"""
        B_wx = S_wx.reshape((len(S_wx), -1))
        nw, nx = B_wx.shape
        tmp_wx = np.zeros((nw, min(nx, self.blocksize)), complex)
        for x in range(0, nx, self.blocksize):
            b_wx = B_wx[:, x:x + self.blocksize]
            c_wx = tmp_wx[:, :b_wx.shape[1]]
            gemm(1.0, b_wx, self.H_ww, 0.0, c_wx)
            b_wx[:] = c_wx


if __name__ == '__main__':
    do = 0.025
    eta = 0.1
    omega_w = frequency_grid(do, 10.0, 3)
    print(len(omega_w))
    X_w = omega_w * 0j
    Xt_w = omega_w * 0j
    Xh_w = omega_w * 0j
    for o in -np.linspace(2.5, 2.9, 10):
        X_w += (1 / (omega_w + o + 1j * eta) -
                1 / (omega_w - o + 1j * eta)) / o**2
        Xt_w += (1 / (omega_w + o - 1j * eta) -
                 1 / (omega_w - o + 1j * eta)) / o**2
        w = int(-o / do / (1 + 3 * -o / 10))
        o1, o2 = omega_w[w:w + 2]
        assert o1 - 1e-12 <= -o <= o2 + 1e-12, (o1, -o, o2)
        p = 1 / (o2 - o1)**2 / o**2
        Xh_w[w] += p * (o2 - -o)
        Xh_w[w + 1] += p * (-o - o1)

    ht = HilbertTransform(omega_w, eta, 1)
    ht(Xh_w)

    import matplotlib.pyplot as plt
    plt.plot(omega_w, X_w.imag, label='ImX')
    plt.plot(omega_w, X_w.real, label='ReX')
    plt.plot(omega_w, Xt_w.imag, label='ImXt')
    plt.plot(omega_w, Xt_w.real, label='ReXt')
    plt.plot(omega_w, Xh_w.imag, label='ImXh')
    plt.plot(omega_w, Xh_w.real, label='ReXh')
    plt.legend()
    plt.show()
