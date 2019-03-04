from __future__ import print_function, division

import functools
import os
from time import ctime

import numpy as np
from ase.units import Hartree
from ase.utils import convert_string_to_fd
from ase.utils.timing import timer, Timer
from scipy.special.orthogonal import p_roots

import gpaw.mpi as mpi
from gpaw import GPAW
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.response.chi0 import Chi0
from gpaw.response.kernels import get_coulomb_kernel
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb
from gpaw.wavefunctions.pw import PWDescriptor, count_reciprocal_vectors


def rpa(filename, ecut=200.0, blocks=1, extrapolate=4):
    """Calculate RPA energy.

    filename: str
        Name of restart-file.
    ecut: float
        Plane-wave cutoff.
    blocks: int
        Split polarizability matrix in this many blocks.
    extrapolate: int
        Number of cutoff energies to use for extrapolation.
    """
    name, ext = filename.rsplit('.', 1)
    assert ext == 'gpw'
    from gpaw.xc.rpa import RPACorrelation
    rpa = RPACorrelation(name, name + '-rpa.dat',
                         nblocks=blocks,
                         txt=name + '-rpa.txt')
    rpa.calculate(ecut=ecut * (1 + 0.5 * np.arange(extrapolate))**(-2 / 3))


class RPACorrelation:
    def __init__(self, calc, xc='RPA', filename=None,
                 skip_gamma=False, qsym=True, nlambda=None,
                 nfrequencies=16, frequency_max=800.0, frequency_scale=2.0,
                 frequencies=None, weights=None, truncation=None,
                 world=mpi.world, nblocks=1, txt='-'):
        """Creates the RPACorrelation object

        calc: str or calculator object
            The string should refer to the .gpw file contaning KS orbitals
        xc: str
            Exchange-correlation kernel. This is only different from RPA when
            this object is constructed from a different module - e.g. fxc.py
        filename: str
            txt output
        skip_gamme: bool
            If True, skip q = [0,0,0] from the calculation
        qsym: bool
            Use symmetry to reduce q-points
        nlambda: int
            Number of lambda points. Only used for numerical coupling
            constant integration involved when called from fxc.py
        nfrequencies: int
            Number of frequency points used in the Gauss-Legendre integration
        frequency_max: float
            Largest frequency point in Gauss-Legendre integration
        frequency_scale: float
            Determines density of frequency points at low frequencies. A slight
            increase to e.g. 2.5 or 3.0 improves convergence wth respect to
            frequency points for metals
        frequencies: list
            List of frequancies for user-specified frequency integration
        weights: list
            list of weights (integration measure) for a user specified
            frequency grid. Must be specified and have the same length as
            frequencies if frequencies is not None
        truncation: str
            Coulomb truncation scheme. Can be either wigner-seitz,
            2D, 1D, or 0D
        world: communicator
        nblocks: int
            Number of parallelization blocks. Frequency parallelization
            can be specified by setting nblocks=nfrequencies and is useful
            for memory consuming calculations
        txt: str
            txt file for saving and loading contributions to the correlation
            energy from different q-points
        """

        if isinstance(calc, str):
            calc = GPAW(calc, txt=None, communicator=mpi.serial_comm)
        self.calc = calc

        self.fd = convert_string_to_fd(txt, world)

        self.timer = Timer()

        if frequencies is None:
            frequencies, weights = get_gauss_legendre_points(nfrequencies,
                                                             frequency_max,
                                                             frequency_scale)
            user_spec = False
        else:
            assert weights is not None
            user_spec = True

        self.omega_w = frequencies / Hartree
        self.weight_w = weights / Hartree

        if nblocks > 1:
            assert len(self.omega_w) % nblocks == 0

        self.nblocks = nblocks
        self.world = world

        self.truncation = truncation
        self.skip_gamma = skip_gamma
        self.ibzq_qc = None
        self.weight_q = None
        self.initialize_q_points(qsym)

        # Energies for all q-vetors and cutoff energies:
        self.energy_qi = []

        self.filename = filename

        self.print_initialization(xc, frequency_scale, nlambda, user_spec)

    def initialize_q_points(self, qsym):
        kd = self.calc.wfs.kd
        self.bzq_qc = kd.get_bz_q_points(first=True)

        if not qsym:
            self.ibzq_qc = self.bzq_qc
            self.weight_q = np.ones(len(self.bzq_qc)) / len(self.bzq_qc)
        else:
            U_scc = kd.symmetry.op_scc
            self.ibzq_qc = kd.get_ibz_q_points(self.bzq_qc, U_scc)[0]
            self.weight_q = kd.q_weights

    def read(self):
        lines = open(self.filename).readlines()[1:]
        n = 0
        self.energy_qi = []
        nq = len(lines) // len(self.ecut_i)
        for q_c in self.ibzq_qc[:nq]:
            self.energy_qi.append([])
            for ecut in self.ecut_i:
                q1, q2, q3, ec, energy = [float(x)
                                          for x in lines[n].split()]
                self.energy_qi[-1].append(energy / Hartree)
                n += 1

                if (abs(q_c - (q1, q2, q3)).max() > 1e-4 or
                    abs(int(ecut * Hartree) - ec) > 0):
                    self.energy_qi = []
                    return

        print('Read %d q-points from file: %s' % (nq, self.filename),
              file=self.fd)
        print(file=self.fd)

    def write(self):
        if self.world.rank == 0 and self.filename:
            fd = open(self.filename, 'w')
            print('#%9s %10s %10s %8s %12s' %
                  ('q1', 'q2', 'q3', 'E_cut', 'E_c(q)'), file=fd)
            for energy_i, q_c in zip(self.energy_qi, self.ibzq_qc):
                for energy, ecut in zip(energy_i, self.ecut_i):
                    print('%10.4f %10.4f %10.4f %8d   %r' %
                          (tuple(q_c) + (ecut * Hartree, energy * Hartree)),
                          file=fd)

    def calculate(self, ecut, nbands=None, spin=False):
        """Calculate RPA correlation energy for one or several cutoffs.

        ecut: float or list of floats
            Plane-wave cutoff(s).
        nbands: int
            Number of bands (defaults to number of plane-waves).
        spin: bool
            Separate spin in response function.
            (Only needed for beyond RPA methods that inherit this function).
        """

        p = functools.partial(print, file=self.fd)

        if isinstance(ecut, (float, int)):
            ecut = ecut * (1 + 0.5 * np.arange(6))**(-2 / 3)
        self.ecut_i = np.asarray(np.sort(ecut)) / Hartree
        ecutmax = max(self.ecut_i)

        if nbands is None:
            p('Response function bands : Equal to number of plane waves')
        else:
            p('Response function bands : %s' % nbands)
        p('Plane wave cutoffs (eV) :', end='')
        for e in self.ecut_i:
            p(' {0:.3f}'.format(e * Hartree), end='')
        p()
        if self.truncation is not None:
            p('Using %s Coulomb truncation' % self.truncation)
        p()

        if self.filename and os.path.isfile(self.filename):
            self.read()
            self.world.barrier()

        chi0 = Chi0(self.calc, frequencies=1j * Hartree * self.omega_w,
                    eta=0.0, intraband=False, hilbert=False,
                    txt='chi0.txt', timer=self.timer, world=self.world,
                    nblocks=self.nblocks)

        self.blockcomm = chi0.blockcomm

        wfs = self.calc.wfs

        if self.truncation == 'wigner-seitz':
            self.wstc = WignerSeitzTruncatedCoulomb(wfs.gd.cell_cv,
                                                    wfs.kd.N_c, self.fd)
        else:
            self.wstc = None

        nq = len(self.energy_qi)
        nw = len(self.omega_w)
        nGmax = max(count_reciprocal_vectors(ecutmax, wfs.gd, q_c)
                    for q_c in self.ibzq_qc[nq:])
        mynGmax = (nGmax + self.nblocks - 1) // self.nblocks

        nx = (1 + spin) * nw * mynGmax * nGmax
        A1_x = np.empty(nx, complex)
        if self.nblocks > 1:
            A2_x = np.empty(nx, complex)
        else:
            A2_x = None

        self.timer.start('RPA')

        for q_c in self.ibzq_qc[nq:]:
            if np.allclose(q_c, 0.0) and self.skip_gamma:
                self.energy_qi.append(len(self.ecut_i) * [0.0])
                self.write()
                p('Not calculating E_c(q) at Gamma')
                p()
                continue

            thisqd = KPointDescriptor([q_c])
            pd = PWDescriptor(ecutmax, wfs.gd, complex, thisqd)
            nG = pd.ngmax
            mynG = (nG + self.nblocks - 1) // self.nblocks
            chi0.Ga = self.blockcomm.rank * mynG
            chi0.Gb = min(chi0.Ga + mynG, nG)

            shape = (1 + spin, nw, chi0.Gb - chi0.Ga, nG)
            chi0_swGG = A1_x[:np.prod(shape)].reshape(shape)
            chi0_swGG[:] = 0.0

            if np.allclose(q_c, 0.0):
                chi0_swxvG = np.zeros((1 + spin, nw, 2, 3, nG), complex)
                chi0_swvv = np.zeros((1 + spin, nw, 3, 3), complex)
            else:
                chi0_swxvG = None
                chi0_swvv = None

            # First not completely filled band:
            m1 = chi0.nocc1
            p('# %s  -  %s' % (len(self.energy_qi), ctime().split()[-2]))
            p('q = [%1.3f %1.3f %1.3f]' % tuple(q_c))

            energy_i = []
            for ecut in self.ecut_i:
                if ecut == ecutmax:
                    # Nothing to cut away:
                    cut_G = None
                    m2 = nbands or nG
                else:
                    cut_G = np.arange(nG)[pd.G2_qG[0] <= 2 * ecut]
                    m2 = len(cut_G)

                p('E_cut = %d eV / Bands = %d:' % (ecut * Hartree, m2))
                self.fd.flush()

                energy = self.calculate_q(chi0, pd,
                                          chi0_swGG, chi0_swxvG, chi0_swvv,
                                          m1, m2, cut_G, A2_x)

                energy_i.append(energy)
                m1 = m2

                a = 1 / chi0.kncomm.size
                if ecut < ecutmax and a != 1.0:
                    # Chi0 will be summed again over chicomm, so we divide
                    # by its size:
                    chi0_swGG *= a
                    #if chi0_swxvG is not None:
                    #    chi0_swxvG *= a
                    #    chi0_swvv *= a

            self.energy_qi.append(energy_i)
            self.write()
            p()

        e_i = np.dot(self.weight_q, np.array(self.energy_qi))
        p('==========================================================')
        p()
        p('Total correlation energy:')
        for e_cut, e in zip(self.ecut_i, e_i):
            p('%6.0f:   %6.4f eV' % (e_cut * Hartree, e * Hartree))
        p()

        self.energy_qi = []  # important if another calculation is performed

        if len(e_i) > 1:
            self.extrapolate(e_i)

        p('Calculation completed at: ', ctime())
        p()

        self.timer.stop('RPA')
        self.timer.write(self.fd)
        self.fd.flush()

        return e_i * Hartree

    @timer('chi0(q)')
    def calculate_q(self, chi0, pd, chi0_swGG, chi0_swxvG, chi0_swvv,
                    m1, m2, cut_G, A2_x):
        chi0_wGG = chi0_swGG[0]
        if chi0_swxvG is not None:
            chi0_wxvG = chi0_swxvG[0]
            chi0_wvv = chi0_swvv[0]
        else:
            chi0_wxvG = None
            chi0_wvv = None
        chi0._calculate(pd, chi0_wGG, chi0_wxvG, chi0_wvv,
                        m1, m2, spins='all', extend_head=False)

        print('E_c(q) = ', end='', file=self.fd)

        chi0_wGG = chi0.redistribute(chi0_wGG, A2_x)

        if not pd.kd.gamma:
            e = self.calculate_energy(pd, chi0_wGG, cut_G)
            print('%.3f eV' % (e * Hartree), file=self.fd)
            self.fd.flush()
        else:
            from ase.dft import monkhorst_pack
            kd = self.calc.wfs.kd
            N = 4
            N_c = np.array([N, N, N])
            if self.truncation is not None:
                N_c[kd.N_c == 1] = 1
            q_qc = monkhorst_pack(N_c) / kd.N_c
            q_qc *= 1.0e-6
            U_scc = kd.symmetry.op_scc
            q_qc = kd.get_ibz_q_points(q_qc, U_scc)[0]
            weight_q = kd.q_weights
            q_qv = 2 * np.pi * np.dot(q_qc, pd.gd.icell_cv)

            nw = len(self.omega_w)
            mynw = nw // self.nblocks
            w1 = self.blockcomm.rank * mynw
            w2 = w1 + mynw
            a_qw = np.sum(np.dot(chi0_wvv[w1:w2], q_qv.T) * q_qv.T, axis=1).T
            a0_qwG = np.dot(q_qv, chi0_wxvG[w1:w2, 0])
            a1_qwG = np.dot(q_qv, chi0_wxvG[w1:w2, 1])

            e = 0
            for iq in range(len(q_qv)):
                chi0_wGG[:, 0] = a0_qwG[iq]
                chi0_wGG[:, :, 0] = a1_qwG[iq]
                chi0_wGG[:, 0, 0] = a_qw[iq]
                ev = self.calculate_energy(pd, chi0_wGG, cut_G,
                                           q_v=q_qv[iq])
                e += ev * weight_q[iq]
            print('%.3f eV' % (e * Hartree), file=self.fd)
            self.fd.flush()

        return e

    @timer('Energy')
    def calculate_energy(self, pd, chi0_wGG, cut_G, q_v=None):
        """Evaluate correlation energy from chi0."""

        sqrV_G = get_coulomb_kernel(pd, self.calc.wfs.kd.N_c, q_v=q_v,
                                    truncation=self.truncation,
                                    wstc=self.wstc)**0.5
        if cut_G is not None:
            sqrV_G = sqrV_G[cut_G]
        nG = len(sqrV_G)

        e_w = []
        for chi0_GG in chi0_wGG:
            if cut_G is not None:
                chi0_GG = chi0_GG.take(cut_G, 0).take(cut_G, 1)

            e_GG = np.eye(nG) - chi0_GG * sqrV_G * sqrV_G[:, np.newaxis]
            e = np.log(np.linalg.det(e_GG)) + nG - np.trace(e_GG)
            e_w.append(e.real)

        E_w = np.zeros_like(self.omega_w)
        self.blockcomm.all_gather(np.array(e_w), E_w)
        energy = np.dot(E_w, self.weight_w) / (2 * np.pi)
        self.E_w = E_w
        return energy

    def extrapolate(self, e_i):
        print('Extrapolated energies:', file=self.fd)
        ex_i = []
        for i in range(len(e_i) - 1):
            e1, e2 = e_i[i:i + 2]
            x1, x2 = self.ecut_i[i:i + 2]**-1.5
            ex = (e1 * x2 - e2 * x1) / (x2 - x1)
            ex_i.append(ex)

            print('  %4.0f -%4.0f:  %5.3f eV' % (self.ecut_i[i] * Hartree,
                                                 self.ecut_i[i + 1] * Hartree,
                                                 ex * Hartree),
                  file=self.fd)
        print(file=self.fd)
        self.fd.flush()

        return e_i * Hartree

    def print_initialization(self, xc, frequency_scale, nlambda, user_spec):
        p = functools.partial(print, file=self.fd)
        p('----------------------------------------------------------')
        p('Non-self-consistent %s correlation energy' % xc)
        p('----------------------------------------------------------')
        p('Started at:  ', ctime())
        p()
        p('Atoms                          :',
          self.calc.atoms.get_chemical_formula(mode='hill'))
        p('Ground state XC functional     :', self.calc.hamiltonian.xc.name)
        p('Valence electrons              :', self.calc.wfs.setups.nvalence)
        p('Number of bands                :', self.calc.wfs.bd.nbands)
        p('Number of spins                :', self.calc.wfs.nspins)
        p('Number of k-points             :', len(self.calc.wfs.kd.bzk_kc))
        p('Number of irreducible k-points :', len(self.calc.wfs.kd.ibzk_kc))
        p('Number of q-points             :', len(self.bzq_qc))
        p('Number of irreducible q-points :', len(self.ibzq_qc))
        p()
        for q, weight in zip(self.ibzq_qc, self.weight_q):
            p('    q: [%1.4f %1.4f %1.4f] - weight: %1.3f' %
              (q[0], q[1], q[2], weight))
        p()
        p('----------------------------------------------------------')
        p('----------------------------------------------------------')
        p()
        if nlambda is None:
            p('Analytical coupling constant integration')
        else:
            p('Numerical coupling constant integration using', nlambda,
              'Gauss-Legendre points')
        p()
        p('Frequencies')
        if not user_spec:
            p('    Gauss-Legendre integration with %s frequency points' %
              len(self.omega_w))
            p('    Transformed from [0,oo] to [0,1] using e^[-aw^(1/B)]')
            p('    Highest frequency point at %5.1f eV and B=%1.1f' %
              (self.omega_w[-1] * Hartree, frequency_scale))
        else:
            p('    User specified frequency integration with',
              len(self.omega_w), 'frequency points')
        p()
        p('Parallelization')
        p('    Total number of CPUs          : % s' % self.world.size)
        p('    G-vector decomposition        : % s' % self.nblocks)
        p('    K-point/band decomposition    : % s' %
          (self.world.size // self.nblocks))
        p()


def get_gauss_legendre_points(nw=16, frequency_max=800.0, frequency_scale=2.0):
    y_w, weights_w = p_roots(nw)
    y_w = y_w.real
    ys = 0.5 - 0.5 * y_w
    ys = ys[::-1]
    w = (-np.log(1 - ys))**frequency_scale
    w *= frequency_max / w[-1]
    alpha = (-np.log(1 - ys[-1]))**frequency_scale / frequency_max
    transform = (-np.log(1 - ys))**(frequency_scale - 1) \
        / (1 - ys) * frequency_scale / alpha
    return w, weights_w * transform / 2


class CLICommand:
    """Run RPA-correlation calculation."""

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('gpw', metavar='gpw-file')
        add('-e', '--cut-off', type=float, default=100, metavar='ECUT',
            help='Plane-wave cut off energy (eV) for polarization function.')
        add('-b', '--blocks', type=int, default=1, metavar='N',
            help='Split polarization matrix in N blocks.')

    @staticmethod
    def run(args):
        assert args.gpw.endswith('.gpw')
        rpa = RPACorrelation(args.gpw,
                             txt=args.gpw[:-3] + 'rpa.txt',
                             nblocks=args.blocks)
        rpa.calculate([args.cut_off])
