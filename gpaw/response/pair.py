from __future__ import print_function, division

import functools
import numbers
import sys
from math import pi

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from ase.units import Hartree
from ase.utils import convert_string_to_fd
from ase.utils.timing import timer, Timer

import gpaw.mpi as mpi
from gpaw import GPAW
from gpaw.fd_operators import Gradient
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.occupations import FermiDirac
from gpaw.response.math_func import (two_phi_planewave_integrals,
                                     two_phi_nabla_planewave_integrals)
from gpaw.utilities.blas import gemm
from gpaw.utilities.progressbar import ProgressBar
from gpaw.wavefunctions.pw import PWLFC
from gpaw.bztools import get_reduced_bz, unique_rows


class KPoint:
    def __init__(self, s, K, n1, n2, blocksize, na, nb,
                 ut_nR, eps_n, f_n, P_ani, shift_c):
        self.s = s    # spin index
        self.K = K    # BZ k-point index
        self.n1 = n1  # first band
        self.n2 = n2  # first band not included
        self.blocksize = blocksize
        self.na = na  # first band of block
        self.nb = nb  # first band of block not included
        self.ut_nR = ut_nR      # periodic part of wave functions in real-space
        self.eps_n = eps_n      # eigenvalues
        self.f_n = f_n          # occupation numbers
        self.P_ani = P_ani      # PAW projections
        self.shift_c = shift_c  # long story - see the
        # PairDensity.construct_symmetry_operators() method


class KPointPair:
    """This class defines the kpoint-pair container object.

    Used for calculating pair quantities it contains two kpoints,
    and an associated set of Fourier components."""
    def __init__(self, kpt1, kpt2, Q_G):
        self.kpt1 = kpt1
        self.kpt2 = kpt2
        self.Q_G = Q_G

    def get_k1(self):
        """ Return KPoint object 1."""
        return self.kpt1

    def get_k2(self):
        """ Return KPoint object 2."""
        return self.kpt2

    def get_planewave_indices(self):
        """ Return the planewave indices associated with this pair."""
        return self.Q_G

    def get_transition_energies(self, n_n, m_m):
        """Return the energy difference for specified bands."""
        n_n = np.array(n_n)
        m_m = np.array(m_m)
        kpt1 = self.kpt1
        kpt2 = self.kpt2
        deps_nm = (kpt1.eps_n[n_n - self.kpt1.n1][:, np.newaxis] -
                   kpt2.eps_n[m_m - self.kpt2.n1])
        return deps_nm

    def get_occupation_differences(self, n_n, m_m):
        """Get difference in occupation factor between specified bands."""
        n_n = np.array(n_n)
        m_m = np.array(m_m)
        kpt1 = self.kpt1
        kpt2 = self.kpt2
        df_nm = (kpt1.f_n[n_n - self.kpt1.n1][:, np.newaxis] -
                 kpt2.f_n[m_m - self.kpt2.n1])
        return df_nm


class PWSymmetryAnalyzer:
    """Class for handling planewave symmetries."""
    def __init__(self, kd, pd, txt=sys.stdout,
                 disable_point_group=False,
                 disable_non_symmorphic=True,
                 disable_time_reversal=False,
                 timer=None):
        """Creates a PWSymmetryAnalyzer object.

        Determines which of the symmetries of the atomic structure
        that is compatible with the reciprocal lattice. Contains the
        necessary functions for mapping quantities between kpoints,
        and or symmetrizing arrays.

        kd: KPointDescriptor
            The kpoint descriptor containing the
            information about symmetries and kpoints.
        pd: PWDescriptor
            Plane wave descriptor that contains the reciprocal
            lattice .
        txt: str
            Output file.
        disable_point_group: bool
            Switch for disabling point group symmetries.
        disable_non_symmorphic:
            Switch for disabling non symmorphic symmetries.
        disable_time_reversal:
            Switch for disabling time reversal.
        """
        self.pd = pd
        self.kd = kd
        self.fd = txt

        # Caveats
        assert disable_non_symmorphic, \
            print('You are not allowed to use non symmorphic syms, sorry. ',
                  file=self.fd)

        # Settings
        self.disable_point_group = disable_point_group
        self.disable_time_reversal = disable_time_reversal
        self.disable_non_symmorphic = disable_non_symmorphic
        if (kd.symmetry.has_inversion or not kd.symmetry.time_reversal) and \
           not self.disable_time_reversal:
            print('\nThe ground calculation does not support time-reversal ' +
                  'symmetry possibly because it has an inversion center ' +
                  'or that it has been manually deactivated. \n', file=self.fd)
            self.disable_time_reversal = True

        self.disable_symmetries = (self.disable_point_group and
                                   self.disable_time_reversal and
                                   self.disable_non_symmorphic)

        # Number of symmetries
        U_scc = kd.symmetry.op_scc
        self.nU = len(U_scc)

        self.nsym = 2 * self.nU
        self.use_time_reversal = not self.disable_time_reversal

        # Which timer to use
        self.timer = timer or Timer()

        self.KDTree = cKDTree(np.mod(np.mod(kd.bzk_kc, 1).round(6), 1))

        # Initialize
        self.initialize()

    @timer('Initialize')
    def initialize(self):
        """Initialize relevant quantities."""
        self.infostring = ''
        if self.disable_point_group:
            self.infostring += 'Point group not included. '
        else:
            self.infostring += 'Point group included. '

        if self.disable_time_reversal:
            self.infostring += 'Time reversal not included. '
        else:
            self.infostring += 'Time reversal included. '

        if self.disable_non_symmorphic:
            self.infostring += 'Disabled non symmorphic symmetries. '
        else:
            self.infostring += 'Time reversal included. '

        if self.disable_symmetries:
            self.infostring += 'All symmetries have been disabled. '

        # Do the work
        self.analyze_symmetries()
        self.analyze_kpoints()
        self.initialize_G_maps()

        # Print info
        print(self.infostring, file=self.fd)
        self.print_symmetries()

    def find_kpoint(self, k_c):
        return self.KDTree.query(np.mod(np.mod(k_c, 1).round(6), 1))[1]

    def print_symmetries(self):
        """Handsome print function for symmetry operations."""

        p = functools.partial(print, file=self.fd)

        p()
        nx = 6 if self.disable_non_symmorphic else 3
        ns = len(self.s_s)
        y = 0
        for y in range((ns + nx - 1) // nx):
            for c in range(3):
                for x in range(nx):
                    s = x + y * nx
                    if s == ns:
                        break
                    tmp = self.get_symmetry_operator(self.s_s[s])
                    op_cc, sign, TR, shift_c, ft_c = tmp
                    op_c = sign * op_cc[c]
                    p('  (%2d %2d %2d)' % tuple(op_c), end='')
                p()
            p()

    @timer('Analyze')
    def analyze_kpoints(self):
        """Calculate the reduction in the number of kpoints."""
        K_gK = self.group_kpoints()
        ng = len(K_gK)
        self.infostring += '{0} groups of equivalent kpoints. '.format(ng)
        percent = (1. - (ng + 0.) / self.kd.nbzkpts) * 100
        self.infostring += '{0}% reduction. '.format(percent)

    @timer('Analyze symmetries.')
    def analyze_symmetries(self):
        """Determine allowed symmetries.

        An direct symmetry U must fulfill::

          U \mathbf{q} = q + \Delta

        Under time-reversal (indirect) it must fulfill::

          -U \mathbf{q} = q + \Delta

        where :math:`\Delta` is a reciprocal lattice vector.
        """
        pd = self.pd

        # Shortcuts
        q_c = pd.kd.bzk_kc[0]
        kd = self.kd

        U_scc = kd.symmetry.op_scc
        nU = self.nU
        nsym = self.nsym

        shift_sc = np.zeros((nsym, 3), int)
        conserveq_s = np.zeros(nsym, bool)

        newq_sc = np.dot(U_scc, q_c)

        # Direct symmetries
        dshift_sc = (newq_sc - q_c[np.newaxis]).round().astype(int)
        inds_s = np.argwhere((newq_sc == q_c[np.newaxis] + dshift_sc).all(1))
        conserveq_s[inds_s] = True

        shift_sc[:nU] = dshift_sc

        # Time reversal
        trshift_sc = (-newq_sc - q_c[np.newaxis]).round().astype(int)
        trinds_s = np.argwhere((-newq_sc == q_c[np.newaxis] +
                                trshift_sc).all(1)) + nU
        conserveq_s[trinds_s] = True
        shift_sc[nU:nsym] = trshift_sc

        # The indices of the allowed symmetries
        s_s = conserveq_s.nonzero()[0]

        # Filter out disabled symmetries
        if self.disable_point_group:
            s_s = list(filter(self.is_not_point_group, s_s))

        if self.disable_time_reversal:
            s_s = list(filter(self.is_not_time_reversal, s_s))

        if self.disable_non_symmorphic:
            s_s = list(filter(self.is_not_non_symmorphic, s_s))

#        stmp_s = []
#        for s in s_s:
#            if self.kd.bz2bz_ks[0, s] == -1:
#                assert (self.kd.bz2bz_ks[:, s] == -1).all()
#            else:
#                stmp_s.append(s)

#        s_s = stmp_s

        self.infostring += 'Found {} allowed symmetries. '.format(len(s_s))
        self.s_s = s_s
        self.shift_sc = shift_sc

    def is_not_point_group(self, s):
        U_scc = self.kd.symmetry.op_scc
        nU = self.nU
        return (U_scc[s % nU] == np.eye(3)).all()

    def is_not_time_reversal(self, s):
        nU = self.nU
        return not bool(s // nU)

    def is_not_non_symmorphic(self, s):
        ft_sc = self.kd.symmetry.ft_sc
        nU = self.nU
        return not bool(ft_sc[s % nU].any())

    def how_many_symmetries(self):
        """Return number of symmetries."""
        return len(self.s_s)

    @timer('Group kpoints')
    def group_kpoints(self, K_k=None):
        """Group kpoints according to the reduced symmetries"""
        if K_k is None:
            K_k = np.arange(self.kd.nbzkpts)
        s_s = self.s_s
        bz2bz_ks = self.kd.bz2bz_ks
        nk = len(bz2bz_ks)
        sbz2sbz_ks = bz2bz_ks[K_k][:, s_s]  # Reduced number of symmetries
        # Avoid -1 (see documentation in gpaw.symmetry)
        sbz2sbz_ks[sbz2sbz_ks == -1] = nk

        smallestk_k = np.sort(sbz2sbz_ks)[:, 0]
        k2g_g = np.unique(smallestk_k, return_index=True)[1]

        K_gs = sbz2sbz_ks[k2g_g]
        K_gk = [np.unique(K_s[K_s != nk]) for K_s in K_gs]

        return K_gk

    def get_BZ(self):
        # Get the little group of q
        U_scc = []
        for s in self.s_s:
            U_cc, sign, _, _, _ = self.get_symmetry_operator(s)
            U_scc.append(sign * U_cc)
        U_scc = np.array(U_scc)

        # Determine the irreducible BZ
        bzk_kc, ibzk_kc = get_reduced_bz(self.pd.gd.cell_cv,
                                         U_scc,
                                         False)

        return bzk_kc

    def get_reduced_kd(self, pbc_c=np.ones(3, bool)):
        # Get the little group of q
        U_scc = []
        for s in self.s_s:
            U_cc, sign, _, _, _ = self.get_symmetry_operator(s)
            U_scc.append(sign * U_cc)
        U_scc = np.array(U_scc)

        # Determine the irreducible BZ
        bzk_kc, ibzk_kc = get_reduced_bz(self.pd.gd.cell_cv,
                                         U_scc,
                                         False,
                                         pbc_c=pbc_c)

        n = 3
        N_xc = np.indices((n, n, n)).reshape((3, n**3)).T - n // 2

        # Find the irreducible kpoints
        tess = Delaunay(ibzk_kc)
        ik_kc = []
        for N_c in N_xc:
            k_kc = self.kd.bzk_kc + N_c
            k_kc = k_kc[tess.find_simplex(k_kc) >= 0]
            if not len(ik_kc) and len(k_kc):
                ik_kc = unique_rows(k_kc)
            elif len(k_kc):
                ik_kc = unique_rows(np.append(k_kc, ik_kc, axis=0))

        return KPointDescriptor(ik_kc)

    def unfold_kpoints(self, points_pv, tol=1e-8, mod=None):
        points_pc = np.dot(points_pv, self.pd.gd.cell_cv.T) / (2 * np.pi)

        # Get the little group of q
        U_scc = []
        for s in self.s_s:
            U_cc, sign, _, _, _ = self.get_symmetry_operator(s)
            U_scc.append(sign * U_cc)
        U_scc = np.array(U_scc)

        points = np.concatenate(np.dot(points_pc, U_scc.transpose(0, 2, 1)))
        points = unique_rows(points, tol=tol, mod=mod)
        points = np.dot(points, self.pd.gd.icell_cv) * (2 * np.pi)
        return points

    def get_kpoint_weight(self, k_c):
        K = self.find_kpoint(k_c)
        iK = self.kd.bz2ibz_k[K]
        K_k = self.unfold_ibz_kpoint(iK)
        K_gK = self.group_kpoints(K_k)

        for K_k in K_gK:
            if K in K_k:
                if self.kd.refine_info is not None:
                    weight = sum(self.kd.refine_info.weight_k[K_k])
                    return weight
                else:
                    return len(K_k)

    def get_kpoint_mapping(self, K1, K2):
        """Get index of symmetry for mapping between K1 and K2"""
        s_s = self.s_s
        bz2bz_ks = self.kd.bz2bz_ks
        bzk2rbz_s = bz2bz_ks[K1][s_s]
        try:
            s = np.argwhere(bzk2rbz_s == K2)[0][0]
        except IndexError:
            print('K = {0} cannot be mapped into K = {1}'.format(K1, K2),
                  file=self.fd)
            raise
        return s_s[s]

    def get_shift(self, K1, K2, U_cc, sign):
        """Get shift for mapping between K1 and K2."""
        kd = self.kd
        k1_c = kd.bzk_kc[K1]
        k2_c = kd.bzk_kc[K2]

        shift_c = np.dot(U_cc, k1_c) - k2_c * sign
        assert np.allclose(shift_c.round(), shift_c)
        shift_c = shift_c.round().astype(int)

        return shift_c

    @timer('map_G')
    def map_G(self, K1, K2, a_MG):
        """Map a function of G from K1 to K2. """
        if len(a_MG) == 0:
            return []

        if K1 == K2:
            return a_MG

        G_G, sign = self.map_G_vectors(K1, K2)

        s = self.get_kpoint_mapping(K1, K2)
        U_cc, _, TR, shift_c, ft_c = self.get_symmetry_operator(s)

        return TR(a_MG[..., G_G])

    def symmetrize_wGG(self, A_wGG):
        """Symmetrize an array in GG'."""

        for A_GG in A_wGG:
            tmp_GG = np.zeros_like(A_GG)

            for s in self.s_s:
                G_G, sign, _ = self.G_sG[s]
                if sign == 1:
                    tmp_GG += A_GG[G_G, :][:, G_G]
                if sign == -1:
                    tmp_GG += A_GG[G_G, :][:, G_G].T

            A_GG[:] = tmp_GG / self.how_many_symmetries()

    def symmetrize_wxx(self, A_wxx, optical_limit=False):
        """Symmetrize an array in xx'."""
        tmp_wxx = np.zeros_like(A_wxx)

        A_cv = self.pd.gd.cell_cv
        iA_cv = self.pd.gd.icell_cv

        if self.use_time_reversal:
            AT_wxx = np.transpose(A_wxx, (0, 2, 1))

        for s in self.s_s:
            G_G, sign, shift_c = self.G_sG[s]
            if optical_limit:
                G_G = np.array(G_G) + 2
                G_G = np.insert(G_G, 0, [0, 1])
                U_cc, _, TR, shift_c, ft_c = self.get_symmetry_operator(s)
                M_vv = np.dot(np.dot(A_cv.T, U_cc.T), iA_cv)

            if sign == 1:
                tmp = A_wxx[:, G_G, :][:, :, G_G]
                if optical_limit:
                    tmp[:, 0:3, :] = np.transpose(np.dot(M_vv.T,
                                                         tmp[:, 0:3, :]),
                                                  (1, 0, 2))
                    tmp[:, :, 0:3] = np.dot(tmp[..., 0:3], M_vv)
                tmp_wxx += tmp
            elif sign == -1:
                tmp = AT_wxx[:, G_G, :][:, :, G_G]
                if optical_limit:
                    tmp[:, 0:3, :] = np.transpose(np.dot(M_vv.T,
                                                         tmp[:, 0:3, :]),
                                                  (1, 0, 2)) * sign
                    tmp[:, :, 0:3] = np.dot(tmp[:, :, 0:3], M_vv) * sign
                tmp_wxx += tmp

        # Inplace overwriting
        A_wxx[:] = tmp_wxx / self.how_many_symmetries()

    def symmetrize_wxvG(self, A_wxvG):
        """Symmetrize chi0_wxvG"""
        A_cv = self.pd.gd.cell_cv
        iA_cv = self.pd.gd.icell_cv

        if self.use_time_reversal:
            # ::-1 corresponds to transpose in wing indices
            AT_wxvG = A_wxvG[:, ::-1]

        tmp_wxvG = np.zeros_like(A_wxvG)
        for s in self.s_s:
            G_G, sign, shift_c = self.G_sG[s]
            U_cc, _, TR, shift_c, ft_c = self.get_symmetry_operator(s)
            M_vv = np.dot(np.dot(A_cv.T, U_cc.T), iA_cv)
            if sign == 1:
                tmp = sign * np.dot(M_vv.T, A_wxvG[..., G_G])
            elif sign == -1:
                tmp = sign * np.dot(M_vv.T, AT_wxvG[..., G_G])
            tmp_wxvG += np.transpose(tmp, (1, 2, 0, 3))

        # Overwrite the input
        A_wxvG[:] = tmp_wxvG / self.how_many_symmetries()

    def symmetrize_wvv(self, A_wvv):
        """Symmetrize chi_wvv."""
        A_cv = self.pd.gd.cell_cv
        iA_cv = self.pd.gd.icell_cv
        tmp_wvv = np.zeros_like(A_wvv)
        if self.use_time_reversal:
            AT_wvv = np.transpose(A_wvv, (0, 2, 1))

        for s in self.s_s:
            G_G, sign, shift_c = self.G_sG[s]
            U_cc, _, TR, shift_c, ft_c = self.get_symmetry_operator(s)
            M_vv = np.dot(np.dot(A_cv.T, U_cc.T), iA_cv)
            if sign == 1:
                tmp = np.dot(np.dot(M_vv.T, A_wvv), M_vv)
            elif sign == -1:
                tmp = np.dot(np.dot(M_vv.T, AT_wvv), M_vv)
            tmp_wvv += np.transpose(tmp, (1, 0, 2))

        # Overwrite the input
        A_wvv[:] = tmp_wvv / self.how_many_symmetries()

    @timer('map_v')
    def map_v(self, K1, K2, a_Mv):
        """Map a function of v (cartesian component) from K1 to K2."""

        if len(a_Mv) == 0:
            return []

        if K1 == K2:
            return a_Mv

        A_cv = self.pd.gd.cell_cv
        iA_cv = self.pd.gd.icell_cv

        # Get symmetry
        s = self.get_kpoint_mapping(K1, K2)
        U_cc, sign, TR, _, ft_c = self.get_symmetry_operator(s)

        # Create cartesian operator
        M_vv = np.dot(np.dot(A_cv.T, U_cc.T), iA_cv)
        return sign * np.dot(TR(a_Mv), M_vv)

    def timereversal(self, s):
        """Is this a time-reversal symmetry?"""
        tr = bool(s // self.nU)
        return tr

    def get_symmetry_operator(self, s):
        """Return symmetry operator s."""
        U_scc = self.kd.symmetry.op_scc
        ft_sc = self.kd.symmetry.op_scc

        reds = s % self.nU
        if self.timereversal(s):
            TR = lambda x: x.conj()
            sign = -1
        else:
            sign = 1
            TR = lambda x: x

        return U_scc[reds], sign, TR, self.shift_sc[s], ft_sc[reds]

    @timer('map_G_vectors')
    def map_G_vectors(self, K1, K2):
        """Return G vector mapping."""
        s = self.get_kpoint_mapping(K1, K2)
        G_G, sign, shift_c = self.G_sG[s]

        return G_G, sign

    def initialize_G_maps(self):
        """Calculate the Gvector mappings."""
        pd = self.pd
        B_cv = 2.0 * np.pi * pd.gd.icell_cv
        G_Gv = pd.get_reciprocal_vectors(add_q=False)
        G_Gc = np.dot(G_Gv, np.linalg.inv(B_cv))
        Q_G = pd.Q_qG[0]

        G_sG = [None] * self.nsym
        UG_sGc = [None] * self.nsym
        Q_sG = [None] * self.nsym
        for s in self.s_s:
            U_cc, sign, TR, shift_c, ft_c = self.get_symmetry_operator(s)
            iU_cc = np.linalg.inv(U_cc).T
            UG_Gc = np.dot(G_Gc - shift_c, sign * iU_cc)

            assert np.allclose(UG_Gc.round(), UG_Gc)
            UQ_G = np.ravel_multi_index(UG_Gc.round().astype(int).T,
                                        pd.gd.N_c, 'wrap')

            G_G = len(Q_G) * [None]
            for G, UQ in enumerate(UQ_G):
                try:
                    G_G[G] = np.argwhere(Q_G == UQ)[0][0]
                except IndexError:
                    print('This should not be possible but' +
                          'a G-vector was mapped outside the sphere')
                    raise IndexError
            UG_sGc[s] = UG_Gc
            Q_sG[s] = UQ_G
            G_sG[s] = [G_G, sign, shift_c]
        self.G_Gc = G_Gc
        self.UG_sGc = UG_sGc
        self.Q_sG = Q_sG
        self.G_sG = G_sG

    def unfold_ibz_kpoint(self, ik):
        """Return kpoints related to irreducible kpoint."""
        kd = self.kd
        K_k = np.unique(kd.bz2bz_ks[kd.ibz2bz_k[ik]])
        K_k = K_k[K_k != -1]
        return K_k


class PairDensity:
    def __init__(self, calc, ecut=50, response='density',
                 ftol=1e-6, threshold=1,
                 real_space_derivatives=False,
                 world=mpi.world, txt='-', timer=None,
                 nblocks=1, gate_voltage=None):

        if ecut is not None:
            ecut /= Hartree

        if gate_voltage is not None:
            gate_voltage = gate_voltage / Hartree
        
        self.response = response
        self.ecut = ecut
        self.ftol = ftol
        self.threshold = threshold
        self.real_space_derivatives = real_space_derivatives
        self.gate_voltage = gate_voltage

        self.timer = timer or Timer()

        self.world = world

        if nblocks == 1:
            self.blockcomm = world.new_communicator([world.rank])
            self.kncomm = world
        else:
            assert world.size % nblocks == 0, world.size
            rank1 = world.rank // nblocks * nblocks
            rank2 = rank1 + nblocks
            self.blockcomm = self.world.new_communicator(range(rank1, rank2))
            ranks = range(world.rank % nblocks, world.size, nblocks)
            self.kncomm = self.world.new_communicator(ranks)

        self.fd = convert_string_to_fd(txt, world)

        with self.timer('Read ground state'):
            if isinstance(calc, str):
                print('Reading ground state calculation:\n  %s' % calc,
                      file=self.fd)
                calc = GPAW(calc, txt=None, communicator=mpi.serial_comm)
            else:
                assert calc.wfs.world.size == 1

        assert calc.wfs.kd.symmetry.symmorphic
        self.calc = calc

        self.fermi_level = self.calc.occupations.get_fermi_level()
        
        if gate_voltage is not None:
            self.add_gate_voltage(gate_voltage)
        else:
            self.add_gate_voltage(gate_voltage=0)

        self.spos_ac = calc.spos_ac

        self.nocc1 = None  # number of completely filled bands
        self.nocc2 = None  # number of non-empty bands
        self.count_occupied_bands()

        self.ut_sKnvR = None  # gradient of wave functions for optical limit

        self.vol = abs(np.linalg.det(calc.wfs.gd.cell_cv))

        kd = self.calc.wfs.kd
        self.KDTree = cKDTree(np.mod(np.mod(kd.bzk_kc, 1).round(6), 1))
        print('Number of blocks:', nblocks, file=self.fd)

    def find_kpoint(self, k_c):
        return self.KDTree.query(np.mod(np.mod(k_c, 1).round(6), 1))[1]

    def add_gate_voltage(self, gate_voltage=0):
        """Shifts the Fermi-level by e * Vg. By definition e = 1."""
        assert isinstance(self.calc.occupations, FermiDirac)
        print('Shifting Fermi-level by %.2f eV' % (gate_voltage * Hartree),
              file=self.fd)
        self.fermi_level += gate_voltage
        for kpt in self.calc.wfs.kpt_u:
            kpt.f_n = (self.shift_occupations(kpt.eps_n, gate_voltage) *
                       kpt.weight)

    def shift_occupations(self, eps_n, gate_voltage):
        """Shift fermilevel."""
        fermi = self.fermi_level
        width = self.calc.occupations.width
        if width < 1e-9:
            return (eps_n < fermi).astype(float)
        else:
            tmp = (eps_n - fermi) / width
        f_n = np.zeros_like(eps_n)
        f_n[tmp <= 100] = 1 / (1 + np.exp(tmp[tmp <= 100]))
        f_n[tmp > 100] = 0.0
        return f_n

    def count_occupied_bands(self):
        self.nocc1 = 9999999
        self.nocc2 = 0
        for kpt in self.calc.wfs.kpt_u:
            f_n = kpt.f_n / kpt.weight
            self.nocc1 = min((f_n > 1 - self.ftol).sum(), self.nocc1)
            self.nocc2 = max((f_n > self.ftol).sum(), self.nocc2)
        print('Number of completely filled bands:', self.nocc1, file=self.fd)
        print('Number of partially filled bands:', self.nocc2, file=self.fd)
        print('Total number of bands:', self.calc.wfs.bd.nbands,
              file=self.fd)

    def distribute_k_points_and_bands(self, band1, band2, kpts=None):
        """Distribute spins, k-points and bands.

        nbands: int
            Number of bands for each spin/k-point combination.

        The attribute self.mysKn1n2 will be set to a list of (s, K, n1, n2)
        tuples that this process handles.
        """

        wfs = self.calc.wfs

        if kpts is None:
            kpts = np.arange(wfs.kd.nbzkpts)

        nbands = band2 - band1
        size = self.kncomm.size
        rank = self.kncomm.rank
        ns = wfs.nspins
        nk = len(kpts)
        n = (ns * nk * nbands + size - 1) // size
        i1 = rank * n
        i2 = min(i1 + n, ns * nk * nbands)

        self.mysKn1n2 = []
        i = 0
        for s in range(ns):
            for K in kpts:
                n1 = min(max(0, i1 - i), nbands)
                n2 = min(max(0, i2 - i), nbands)
                if n1 != n2:
                    self.mysKn1n2.append((s, K, n1 + band1, n2 + band1))
                i += nbands

        print('BZ k-points:', self.calc.wfs.kd, file=self.fd)
        print('Distributing spins, k-points and bands (%d x %d x %d)' %
              (ns, nk, nbands),
              'over %d process%s' %
              (self.kncomm.size, ['es', ''][self.kncomm.size == 1]),
              file=self.fd)
        print('Number of blocks:', self.blockcomm.size, file=self.fd)

    @timer('Get a k-point')
    def get_k_point(self, s, k_c, n1, n2, load_wfs=True, block=False):
        """Return wave functions for a specific k-point and spin.

        s: int
            Spin index (0 or 1).
        K: int
            BZ k-point index.
        n1, n2: int
            Range of bands to include.
        """

        wfs = self.calc.wfs
        kd = wfs.kd

        # Parse kpoint: is k_c an index or a vector
        if not isinstance(k_c, numbers.Integral):
            K = self.find_kpoint(k_c)
            shift0_c = (kd.bzk_kc[K] - k_c).round().astype(int)
        else:
            # Fall back to index
            K = k_c
            shift0_c = np.array([0, 0, 0])
            k_c = None

        if block:
            nblocks = self.blockcomm.size
            rank = self.blockcomm.rank
        else:
            nblocks = 1
            rank = 0

        blocksize = (n2 - n1 + nblocks - 1) // nblocks
        na = min(n1 + rank * blocksize, n2)
        nb = min(na + blocksize, n2)

        U_cc, T, a_a, U_aii, shift_c, time_reversal = \
            self.construct_symmetry_operators(K, k_c=k_c)

        shift_c += -shift0_c
        ik = wfs.kd.bz2ibz_k[K]
        kpt = wfs.kpt_u[s * wfs.kd.nibzkpts + ik]

        assert n2 <= len(kpt.eps_n), \
            'Increase GS-nbands or decrease chi0-nbands!'
        eps_n = kpt.eps_n[n1:n2]
        f_n = kpt.f_n[n1:n2] / kpt.weight
        
        if not load_wfs:
            return KPoint(s, K, n1, n2, blocksize, na, nb,
                          None, eps_n, f_n, None, shift_c)

        with self.timer('load wfs'):
            psit_nG = kpt.psit_nG
            ut_nR = wfs.gd.empty(nb - na, wfs.dtype)
            for n in range(na, nb):
                ut_nR[n - na] = T(wfs.pd.ifft(psit_nG[n], ik))

        with self.timer('Load projections'):
            P_ani = []
            for b, U_ii in zip(a_a, U_aii):
                P_ni = np.dot(kpt.P_ani[b][na:nb], U_ii)
                if time_reversal:
                    P_ni = P_ni.conj()
                P_ani.append(P_ni)

        return KPoint(s, K, n1, n2, blocksize, na, nb,
                      ut_nR, eps_n, f_n, P_ani, shift_c)

    def generate_pair_densities(self, pd, m1, m2, spins, intraband=True,
                                PWSA=None, disable_optical_limit=False,
                                unsymmetrized=False, use_more_memory=1):
        """Generator for returning pair densities.

        Returns the pair densities between the occupied and
        the states in range(m1, m2).

        pd: PWDescriptor
            Plane-wave descriptor for a single q-point.
        m1: int
            Index of first unoccupied band.
        m2: int
            Index of last unoccupied band.
        spins: list
            List of spin indices included.
        intraband: bool
            Include intraband transitions in optical limit.
        PWSA: PlanewaveSymmetryAnalyzer
            If supplied uses this object to determine the symmetries
            of the pair-densities.
        disable_optical_limit: bool
            Disable optical limit.
        unsymmetrized: bool
            Only return pair-densities from one kpoint in each
            group of equivalent kpoints.
        use_more_memory: float
            Group more pair densities for several occupied bands
            together before returning. Here 0 <= use_more_memory <= 1,
            where zero is the minimal amount of memory, and 1 is the maximal.
        """
        assert 0 <= use_more_memory <= 1

        q_c = pd.kd.bzk_kc[0]
        optical_limit = np.allclose(q_c, 0.0) and self.response == 'density'
        optical_limit = not disable_optical_limit and optical_limit

        Q_aGii = self.initialize_paw_corrections(pd)
        self.Q_aGii = Q_aGii  # This is used in g0w0

        if PWSA is None:
            with self.timer('Symmetry analyzer'):
                PWSA = PWSymmetryAnalyzer  # Line too long otherwise
                PWSA = PWSA(self.calc.wfs.kd, pd,
                            timer=self.timer, txt=self.fd)

        pb = ProgressBar(self.fd)
        for kn, (s, ik, n1, n2) in pb.enumerate(self.mysKn1n2):
            Kstar_k = PWSA.unfold_ibz_kpoint(ik)
            for K_k in PWSA.group_kpoints(Kstar_k):
                # Let the first kpoint of the group represent
                # the rest of the kpoints
                K1 = K_k[0]
                # In this way wavefunctions are only loaded into
                # memory for this particular set of kpoints
                kptpair = self.get_kpoint_pair(pd, s, K1, n1, n2, m1, m2)
                kpt1 = kptpair.get_k1()  # kpt1 = k

                if kpt1.s not in spins:
                    continue
                kpt2 = kptpair.get_k2()  # kpt2 = k + q

                if unsymmetrized:
                    # Number of times kpoints are mapped into themselves
                    weight = np.sqrt(PWSA.how_many_symmetries() / len(K_k))

                # Use kpt2 to compute intraband transitions
                # These conditions are sufficient to make sure
                # that it still works in parallel
                if kpt1.n1 == 0 and self.blockcomm.rank == 0 and \
                   optical_limit and intraband:
                    assert self.nocc2 <= kpt2.nb, \
                        print('Error: Too few unoccupied bands')
                    vel0_mv = self.intraband_pair_density(kpt2)
                    f_m = kpt2.f_n[kpt2.na - kpt2.n1:kpt2.nb - kpt2.n1]
                    with self.timer('intraband'):
                        if vel0_mv is not None:
                            if unsymmetrized:
                                yield (f_m, None, None,
                                       None, None, vel0_mv / weight)
                            else:
                                for K2 in K_k:
                                    vel_mv = PWSA.map_v(K1, K2, vel0_mv)
                                    yield (f_m, None, None,
                                           None, None, vel_mv)

                # Divide the occupied bands into chunks
                n_n = np.arange(n2 - n1)
                if use_more_memory == 0:
                    chunksize = 1
                else:
                    chunksize = np.ceil(len(n_n) *
                                        use_more_memory).astype(int)

                no_n = []
                for i in range(len(n_n) // chunksize):
                    i1 = i * chunksize
                    i2 = min((i + 1) * chunksize, len(n_n))
                    no_n.append(n_n[i1:i2])

                # n runs over occupied bands
                for n_n in no_n:  # n_n is a list of occupied band indices
                    # m over unoccupied bands
                    m_m = np.arange(0, kpt2.n2 - kpt2.n1)
                    deps_nm = kptpair.get_transition_energies(n_n, m_m)
                    df_nm = kptpair.get_occupation_differences(n_n, m_m)

                    # This is not quite right for
                    # degenerate partially occupied
                    # bands, but good enough for now:
                    df_nm[df_nm <= 1e-20] = 0.0

                    # Get pair density for representative kpoint
                    ol = optical_limit
                    n0_nmG, n0_nmv, _ = self.get_pair_density(pd, kptpair,
                                                              n_n, m_m,
                                                              optical_limit=ol,
                                                              intraband=False,
                                                              Q_aGii=Q_aGii)

                    n0_nmG[deps_nm >= 0.0] = 0.0
                    if optical_limit:
                        n0_nmv[deps_nm >= 0.0] = 0.0

                    # Reshape nm -> m
                    nG = pd.ngmax
                    deps_m = deps_nm.reshape(-1)
                    df_m = df_nm.reshape(-1)
                    n0_mG = n0_nmG.reshape((-1, nG))
                    if optical_limit:
                        n0_mv = n0_nmv.reshape((-1, 3))

                    if unsymmetrized:
                        if optical_limit:
                            yield (None, df_m, deps_m,
                                   n0_mG / weight, n0_mv / weight, None)
                        else:
                            yield (None, df_m, deps_m,
                                   n0_mG / weight, None, None)
                        continue

                    # Collect pair densities in a single array
                    # and return them
                    nm = n0_mG.shape[0]
                    nG = n0_mG.shape[1]
                    nk = len(K_k)

                    n_MG = np.empty((nm * nk, nG), complex)
                    if optical_limit:
                        n_Mv = np.empty((nm * nk, 3), complex)
                    deps_M = np.tile(deps_m, nk)
                    df_M = np.tile(df_m, nk)

                    for i, K2 in enumerate(K_k):
                        i1 = i * nm
                        i2 = (i + 1) * nm
                        n_mG = PWSA.map_G(K1, K2, n0_mG)

                        if optical_limit:
                            n_mv = PWSA.map_v(K1, K2, n0_mv)
                            n_mG[:, 0] = n_mv[:, 0]
                            n_Mv[i1:i2, :] = n_mv

                        n_MG[i1:i2, :] = n_mG

                    if optical_limit:
                        yield (None, df_M, deps_M, n_MG, n_Mv, None)
                    else:
                        yield (None, df_M, deps_M, n_MG, None, None)

        pb.finish()

    @timer('Get kpoint pair')
    def get_kpoint_pair(self, pd, s, Kork_c, n1, n2, m1, m2,
                        load_wfs=True, block=False):
        # wfs = self.calc.wfs
        # bzk_kc = wfs.kd.bzk_kc

        if isinstance(Kork_c, int):
            # If k_c is an integer then it refers to
            # the index of the kpoint in the BZ
            k_c = self.calc.wfs.kd.bzk_kc[Kork_c]
        else:
            k_c = Kork_c

        q_c = pd.kd.bzk_kc[0]
        with self.timer('get k-points'):
            kpt1 = self.get_k_point(s, k_c, n1, n2, load_wfs=load_wfs)
            # K2 = wfs.kd.find_k_plus_q(q_c, [kpt1.K])[0]
            if self.response in ['+-', '-+']:
                s2 = 1 - s
            else:
                s2 = s
            kpt2 = self.get_k_point(s2, k_c + q_c, m1, m2,
                                    load_wfs=load_wfs, block=block)

        with self.timer('fft indices'):
            Q_G = self.get_fft_indices(kpt1.K, kpt2.K, q_c, pd,
                                       kpt1.shift_c - kpt2.shift_c)

        return KPointPair(kpt1, kpt2, Q_G)

    @timer('get_pair_density')
    def get_pair_density(self, pd, kptpair, n_n, m_m,
                         optical_limit=False, intraband=False,
                         Q_aGii=None, block=False, direction=2,
                         extend_head=True):
        """Get pair density for a kpoint pair."""
        ol = optical_limit = np.allclose(pd.kd.bzk_kc[0], 0.0) and \
            self.response == 'density'
        eh = extend_head
        cpd = self.calculate_pair_densities  # General pair densities
        opd = self.optical_pair_density  # Interband pair densities / q

        if Q_aGii is None:
            Q_aGii = self.initialize_paw_corrections(pd)

        kpt1 = kptpair.kpt1
        kpt2 = kptpair.kpt2
        Q_G = kptpair.Q_G  # Fourier components of kpoint pair
        nG = len(Q_G)

        if extend_head:
            n_nmG = np.zeros((len(n_n), len(m_m), nG + 2 * ol), pd.dtype)
        else:
            n_nmG = np.zeros((len(n_n), len(m_m), nG), pd.dtype)

        for j, n in enumerate(n_n):
            Q_G = kptpair.Q_G
            with self.timer('conj'):
                ut1cc_R = kpt1.ut_nR[n - kpt1.na].conj()
            with self.timer('paw'):
                C1_aGi = [np.dot(Q_Gii, P1_ni[n - kpt1.na].conj())
                          for Q_Gii, P1_ni in zip(Q_aGii, kpt1.P_ani)]
                n_nmG[j, :, 2 * ol * eh:] = cpd(ut1cc_R, C1_aGi, kpt2, pd, Q_G,
                                                block=block)
            if optical_limit:
                if extend_head:
                    n_nmG[j, :, 0:3] = opd(n, m_m, kpt1, kpt2,
                                           block=block)
                else:
                    n_nmG[j, :, 0] = opd(n, m_m, kpt1, kpt2,
                                         block=block)[:, direction]
        return n_nmG

    @timer('get_pair_momentum')
    def get_pair_momentum(self, pd, kptpair, n_n, m_m, Q_avGii=None):
        """Calculate matrix elements of the momentum operator.

        Calculates::

          n_{nm\mathrm{k}}\int_{\Omega_{\mathrm{cell}}}\mathrm{d}\mathbf{r}
          \psi_{n\mathrm{k}}^*(\mathbf{r})
          e^{-i\,(\mathrm{q} + \mathrm{G})\cdot\mathbf{r}}
          \nabla\psi_{m\mathrm{k} + \mathrm{q}}(\mathbf{r})

        pd: PlaneWaveDescriptor
            Plane wave descriptor of a single q_c.
        kptpair: KPointPair
            KpointPair object containing the two kpoints.
        n_n: list
            List of left-band indices (n).
        m_m:
            List of right-band indices (m).
        """
        wfs = self.calc.wfs

        kpt1 = kptpair.kpt1
        kpt2 = kptpair.kpt2
        Q_G = kptpair.Q_G  # Fourier components of kpoint pair

        # For the same band we
        kd = wfs.kd
        gd = wfs.gd
        k_c = kd.bzk_kc[kpt1.K] + kpt1.shift_c
        k_v = 2 * np.pi * np.dot(k_c, np.linalg.inv(gd.cell_cv).T)

        # Calculate k + G
        G_Gv = pd.get_reciprocal_vectors(add_q=True)
        kqG_Gv = k_v[np.newaxis] + G_Gv

        # Pair velocities
        n_nmvG = pd.zeros((len(n_n), len(m_m), 3))

        # Calculate derivatives of left-wavefunction
        # (there will typically be fewer of these)
        ut_nvR = self.make_derivative(kpt1.s, kpt1.K, kpt1.n1, kpt1.n2)

        # PAW-corrections
        if Q_avGii is None:
            Q_avGii = self.initialize_paw_nabla_corrections(pd)

        # Iterate over occupied bands
        for j, n in enumerate(n_n):
            ut1cc_R = kpt1.ut_nR[n].conj()

            n_mG = self.calculate_pair_densities(ut1cc_R,
                                                 [], kpt2,
                                                 pd, Q_G)

            n_nmvG[j] = 1j * kqG_Gv.T[np.newaxis] * n_mG[:, np.newaxis]

            # Treat each cartesian component at a time
            for v in range(3):
                # Minus from integration by parts
                utvcc_R = -ut_nvR[n, v].conj()
                Cv1_aGi = [np.dot(P1_ni[n].conj(), Q_vGii[v])
                           for Q_vGii, P1_ni in zip(Q_avGii, kpt1.P_ani)]

                nv_mG = self.calculate_pair_densities(utvcc_R,
                                                      Cv1_aGi, kpt2,
                                                      pd, Q_G)

                n_nmvG[j, :, v] += nv_mG

        # We want the momentum operator
        n_nmvG *= -1j

        return n_nmvG

    @timer('Calculate pair-densities')
    def calculate_pair_densities(self, ut1cc_R, C1_aGi, kpt2, pd, Q_G,
                                 block=True):
        """Calculate FFT of pair-densities and add PAW corrections.

        ut1cc_R: 3-d complex ndarray
            Complex conjugate of the periodic part of the left hand side
            wave function.
        C1_aGi: list of ndarrays
            PAW corrections for all atoms.
        kpt2: KPoint object
            Right hand side k-point object.
        pd: PWDescriptor
            Plane-wave descriptor for for q=k2-k1.
        Q_G: 1-d int ndarray
            Mapping from flattened 3-d FFT grid to 0.5(G+q)^2<ecut sphere.
        """

        dv = pd.gd.dv
        n_mG = pd.empty(kpt2.blocksize)
        myblocksize = kpt2.nb - kpt2.na

        for ut_R, n_G in zip(kpt2.ut_nR, n_mG):
            n_R = ut1cc_R * ut_R
            with self.timer('fft'):
                n_G[:] = pd.fft(n_R, 0, Q_G) * dv

        # PAW corrections:
        with self.timer('gemm'):
            for C1_Gi, P2_mi in zip(C1_aGi, kpt2.P_ani):
                gemm(1.0, C1_Gi, P2_mi, 1.0, n_mG[:myblocksize], 't')

        if not block or self.blockcomm.size == 1:
            return n_mG
        else:
            n_MG = pd.empty(kpt2.blocksize * self.blockcomm.size)
            self.blockcomm.all_gather(n_mG, n_MG)
            return n_MG[:kpt2.n2 - kpt2.n1]

    @timer('Optical limit')
    def optical_pair_velocity(self, n, m_m, kpt1, kpt2, block=False):
        if self.ut_sKnvR is None or kpt1.K not in self.ut_sKnvR[kpt1.s]:
            self.ut_sKnvR = self.calculate_derivatives(kpt1)

        kd = self.calc.wfs.kd
        gd = self.calc.wfs.gd
        k_c = kd.bzk_kc[kpt1.K] + kpt1.shift_c
        k_v = 2 * np.pi * np.dot(k_c, np.linalg.inv(gd.cell_cv).T)

        ut_vR = self.ut_sKnvR[kpt1.s][kpt1.K][n - kpt1.n1]
        atomdata_a = self.calc.wfs.setups
        C_avi = [np.dot(atomdata.nabla_iiv.T, P_ni[n - kpt1.na])
                 for atomdata, P_ni in zip(atomdata_a, kpt1.P_ani)]

        blockbands = kpt2.nb - kpt2.na
        n0_mv = np.empty((kpt2.blocksize, 3), dtype=complex)
        nt_m = np.empty(kpt2.blocksize, dtype=complex)
        n0_mv[:blockbands] = -self.calc.wfs.gd.integrate(ut_vR,
                                                         kpt2.ut_nR).T
        nt_m[:blockbands] = self.calc.wfs.gd.integrate(kpt1.ut_nR[n - kpt1.na],
                                                       kpt2.ut_nR)

        n0_mv[:blockbands] += (1j * nt_m[:blockbands, np.newaxis] *
                               k_v[np.newaxis, :])

        for C_vi, P_mi in zip(C_avi, kpt2.P_ani):
            gemm(1.0, C_vi, P_mi, 1.0, n0_mv[:blockbands], 'c')

        if block and self.blockcomm.size > 1:
            n0_Mv = np.empty((kpt2.blocksize * self.blockcomm.size, 3),
                             dtype=complex)
            self.blockcomm.all_gather(n0_mv, n0_Mv)
            n0_mv = n0_Mv[:kpt2.n2 - kpt2.n1]

        return -1j * n0_mv

    def optical_pair_density(self, n, m_m, kpt1, kpt2,
                             block=False):
        # Relative threshold for perturbation theory
        threshold = self.threshold

        eps1 = kpt1.eps_n[n - kpt1.n1]
        deps_m = (eps1 - kpt2.eps_n)[m_m - kpt2.n1]
        n0_mv = self.optical_pair_velocity(n, m_m, kpt1, kpt2,
                                           block=block)

        deps_m = deps_m.copy()
        deps_m[deps_m == 0.0] = np.inf

        smallness_mv = np.abs(-1e-3 * n0_mv / deps_m[:, np.newaxis])
        inds_mv = (np.logical_and(np.inf > smallness_mv,
                                  smallness_mv > threshold))
        n0_mv *= - 1 / deps_m[:, np.newaxis]
        n0_mv[inds_mv] = 0

        return n0_mv

    @timer('Intraband')
    def intraband_pair_density(self, kpt, n_n=None,
                               only_partially_occupied=False):
        """Calculate intraband matrix elements of nabla"""
        # Bands and check for block parallelization
        na, nb, n1 = kpt.na, kpt.nb, kpt.n1
        vel_nv = np.zeros((nb - na, 3), dtype=complex)
        if n_n is None:
            n_n = np.arange(na, nb)
        assert np.max(n_n) < nb, print('This is too many bands')
        assert np.min(n_n) >= na, print('This is too few bands')

        # Load kpoints
        kd = self.calc.wfs.kd
        gd = self.calc.wfs.gd
        k_c = kd.bzk_kc[kpt.K] + kpt.shift_c
        k_v = 2 * np.pi * np.dot(k_c, np.linalg.inv(gd.cell_cv).T)
        atomdata_a = self.calc.wfs.setups
        f_n = kpt.f_n

        # No carriers when T=0
        width = self.calc.occupations.width

        # Only works with Fermi-Dirac distribution
        assert isinstance(self.calc.occupations, FermiDirac)
        if width > 1e-15:
            dfde_n = -1 / width * (f_n - f_n**2.0)  # Analytical derivative
            partocc_n = np.abs(dfde_n) > 1e-5  # Is part. occupied?
        else:
            # Just include all bands to be sure
            partocc_n = np.ones(len(f_n), dtype=bool)

        if only_partially_occupied and not partocc_n.any():
            return None

        if only_partially_occupied:
            # Check for block par. consistency
            assert (partocc_n < nb).all(), \
                print('Include more unoccupied bands ', +
                      'or less block parr.', file=self.fd)

        # Break bands into degenerate chunks
        degchunks_cn = []  # indexing c as chunk number
        for n in n_n:
            inds_n = np.nonzero(np.abs(kpt.eps_n[n - n1] -
                                       kpt.eps_n) < 1e-5)[0] + n1

            # Has this chunk already been computed?
            oldchunk = any([n in chunk for chunk in degchunks_cn])
            if not oldchunk and \
               (partocc_n[n - n1] or not only_partially_occupied):
                assert all([ind in n_n for ind in inds_n]), \
                    print('\nYou are cutting over a degenerate band ' +
                          'using block parallelization.',
                          inds_n, n_n, file=self.fd)
                degchunks_cn.append((inds_n))

        # Calculate matrix elements by diagonalizing each block
        for ind_n in degchunks_cn:
            deg = len(ind_n)
            ut_nvR = self.calc.wfs.gd.zeros((deg, 3), complex)
            vel_nnv = np.zeros((deg, deg, 3), dtype=complex)
            # States are included starting from kpt.na
            ut_nR = kpt.ut_nR[ind_n - na]

            # Get derivatives
            for ind, ut_vR in zip(ind_n, ut_nvR):
                ut_vR[:] = self.make_derivative(kpt.s, kpt.K,
                                                ind, ind + 1)[0]

            # Treat the whole degenerate chunk
            for n in range(deg):
                ut_vR = ut_nvR[n]
                C_avi = [np.dot(atomdata.nabla_iiv.T, P_ni[ind_n[n] - na])
                         for atomdata, P_ni in zip(atomdata_a, kpt.P_ani)]

                nabla0_nv = -self.calc.wfs.gd.integrate(ut_vR, ut_nR).T
                nt_n = self.calc.wfs.gd.integrate(ut_nR[n], ut_nR)
                nabla0_nv += 1j * nt_n[:, np.newaxis] * k_v[np.newaxis, :]

                for C_vi, P_ni in zip(C_avi, kpt.P_ani):
                    gemm(1.0, C_vi, P_ni[ind_n - na], 1.0, nabla0_nv, 'c')

                vel_nnv[n] = -1j * nabla0_nv

            for iv in range(3):
                vel, _ = np.linalg.eig(vel_nnv[..., iv])
                vel_nv[ind_n - na, iv] = vel  # Use eigenvalues

        return vel_nv[n_n - na]

    def get_fft_indices(self, K1, K2, q_c, pd, shift0_c):
        """Get indices for G-vectors inside cutoff sphere."""
        kd = self.calc.wfs.kd
        N_G = pd.Q_qG[0]
        shift_c = (shift0_c +
                   (q_c - kd.bzk_kc[K2] + kd.bzk_kc[K1]).round().astype(int))
        if shift_c.any():
            n_cG = np.unravel_index(N_G, pd.gd.N_c)
            n_cG = [n_G + shift for n_G, shift in zip(n_cG, shift_c)]
            N_G = np.ravel_multi_index(n_cG, pd.gd.N_c, 'wrap')
        return N_G

    def construct_symmetry_operators(self, K, k_c=None):
        """Construct symmetry operators for wave function and PAW projections.

        We want to transform a k-point in the irreducible part of the BZ to
        the corresponding k-point with index K.

        Returns U_cc, T, a_a, U_aii, shift_c and time_reversal, where:

        * U_cc is a rotation matrix.
        * T() is a function that transforms the periodic part of the wave
          function.
        * a_a is a list of symmetry related atom indices
        * U_aii is a list of rotation matrices for the PAW projections
        * shift_c is three integers: see code below.
        * time_reversal is a flag - if True, projections should be complex
          conjugated.

        See the get_k_point() method for how to use these tuples.
        """

        wfs = self.calc.wfs
        kd = wfs.kd

        s = kd.sym_k[K]
        U_cc = kd.symmetry.op_scc[s]
        time_reversal = kd.time_reversal_k[K]
        ik = kd.bz2ibz_k[K]
        if k_c is None:
            k_c = kd.bzk_kc[K]
        ik_c = kd.ibzk_kc[ik]

        sign = 1 - 2 * time_reversal
        shift_c = np.dot(U_cc, ik_c) - k_c * sign

        try:
            assert np.allclose(shift_c.round(), shift_c)
        except AssertionError:
            print('shift_c ' + str(shift_c), file=self.fd)
            print('k_c ' + str(k_c), file=self.fd)
            print('kd.bzk_kc[K] ' + str(kd.bzk_kc[K]), file=self.fd)
            print('ik_c ' + str(ik_c), file=self.fd)
            print('U_cc ' + str(U_cc), file=self.fd)
            print('sign ' + str(sign), file=self.fd)
            raise AssertionError

        shift_c = shift_c.round().astype(int)

        if (U_cc == np.eye(3)).all():
            T = lambda f_R: f_R
        else:
            N_c = self.calc.wfs.gd.N_c
            i_cr = np.dot(U_cc.T, np.indices(N_c).reshape((3, -1)))
            i = np.ravel_multi_index(i_cr, N_c, 'wrap')
            T = lambda f_R: f_R.ravel()[i].reshape(N_c)

        if time_reversal:
            T0 = T
            T = lambda f_R: T0(f_R).conj()
            shift_c *= -1

        a_a = []
        U_aii = []
        for a, id in enumerate(self.calc.wfs.setups.id_a):
            b = kd.symmetry.a_sa[s, a]
            S_c = np.dot(self.spos_ac[a], U_cc) - self.spos_ac[b]
            x = np.exp(2j * pi * np.dot(ik_c, S_c))
            U_ii = wfs.setups[a].R_sii[s].T * x
            a_a.append(b)
            U_aii.append(U_ii)

        return U_cc, T, a_a, U_aii, shift_c, time_reversal

    @timer('Initialize PAW corrections')
    def initialize_paw_corrections(self, pd, soft=False):
        wfs = self.calc.wfs
        q_v = pd.K_qv[0]
        optical_limit = np.allclose(q_v, 0) and self.response == 'density'

        G_Gv = pd.get_reciprocal_vectors()
        if optical_limit:
            G_Gv[0] = 1

        pos_av = np.dot(self.spos_ac, pd.gd.cell_cv)

        # Collect integrals for all species:
        Q_xGii = {}
        for id, atomdata in wfs.setups.setups.items():
            if soft:
                ghat = PWLFC([atomdata.ghat_l], pd)
                ghat.set_positions(np.zeros((1, 3)))
                Q_LG = ghat.expand().T
                if atomdata.Delta_iiL is None:
                    ni = atomdata.ni
                    Q_Gii = np.zeros((Q_LG.shape[1], ni, ni))
                else:
                    Q_Gii = np.dot(atomdata.Delta_iiL, Q_LG).T
            else:
                Q_Gii = two_phi_planewave_integrals(G_Gv, atomdata)
                ni = atomdata.ni
                Q_Gii.shape = (-1, ni, ni)

            Q_xGii[id] = Q_Gii

        Q_aGii = []
        for a, atomdata in enumerate(wfs.setups):
            id = wfs.setups.id_a[a]
            Q_Gii = Q_xGii[id]
            x_G = np.exp(-1j * np.dot(G_Gv, pos_av[a]))
            Q_aGii.append(x_G[:, np.newaxis, np.newaxis] * Q_Gii)
            if optical_limit:
                Q_aGii[a][0] = atomdata.dO_ii

        return Q_aGii

    @timer('Initialize PAW corrections')
    def initialize_paw_nabla_corrections(self, pd, soft=False):
        print('Initializing nabla PAW Corrections', file=self.fd)
        wfs = self.calc.wfs
        G_Gv = pd.get_reciprocal_vectors()
        pos_av = np.dot(self.spos_ac, pd.gd.cell_cv)

        # Collect integrals for all species:
        Q_xvGii = {}
        for id, atomdata in wfs.setups.setups.items():
            if soft:
                raise NotImplementedError
            else:
                Q_vGii = two_phi_nabla_planewave_integrals(G_Gv, atomdata)
                ni = atomdata.ni
                Q_vGii.shape = (3, -1, ni, ni)

            Q_xvGii[id] = Q_vGii

        Q_avGii = []
        for a, atomdata in enumerate(wfs.setups):
            id = wfs.setups.id_a[a]
            Q_vGii = Q_xvGii[id]
            x_G = np.exp(-1j * np.dot(G_Gv, pos_av[a]))
            Q_avGii.append(x_G[np.newaxis, :, np.newaxis, np.newaxis] * Q_vGii)

        return Q_avGii

    def calculate_derivatives(self, kpt):
        ut_sKnvR = [{}, {}]
        ut_nvR = self.make_derivative(kpt.s, kpt.K, kpt.n1, kpt.n2)
        ut_sKnvR[kpt.s][kpt.K] = ut_nvR

        return ut_sKnvR

    @timer('Derivatives')
    def make_derivative(self, s, K, n1, n2):
        wfs = self.calc.wfs
        if self.real_space_derivatives:
            grad_v = [Gradient(wfs.gd, v, 1.0, 4, complex).apply
                      for v in range(3)]

        U_cc, T, a_a, U_aii, shift_c, time_reversal = \
            self.construct_symmetry_operators(K)
        A_cv = wfs.gd.cell_cv
        M_vv = np.dot(np.dot(A_cv.T, U_cc.T), np.linalg.inv(A_cv).T)
        ik = wfs.kd.bz2ibz_k[K]
        kpt = wfs.kpt_u[s * wfs.kd.nibzkpts + ik]
        psit_nG = kpt.psit_nG
        iG_Gv = 1j * wfs.pd.get_reciprocal_vectors(q=ik, add_q=False)
        ut_nvR = wfs.gd.zeros((n2 - n1, 3), complex)
        for n in range(n1, n2):
            for v in range(3):
                if self.real_space_derivatives:
                    ut_R = T(wfs.pd.ifft(psit_nG[n], ik))
                    grad_v[v](ut_R, ut_nvR[n - n1, v],
                              np.ones((3, 2), complex))
                else:
                    ut_R = T(wfs.pd.ifft(iG_Gv[:, v] * psit_nG[n], ik))
                    for v2 in range(3):
                        ut_nvR[n - n1, v2] += ut_R * M_vv[v, v2]

        return ut_nvR
