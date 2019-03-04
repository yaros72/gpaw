import numpy as np

from gpaw.utilities.blas import gemm, mmm
from gpaw.utilities import unpack
from gpaw.utilities.partition import EvenPartitioning


def get_atomic_correction(name):
    cls = dict(dense=DenseAtomicCorrection,
               distributed=DistributedAtomicCorrection,
               scipy=ScipyAtomicCorrection)[name]
    return cls()


class BaseAtomicCorrection:
    name = 'base'
    description = 'base class for atomic corrections with LCAO'

    def __init__(self):
        self.nops = 0

    def redistribute(self, wfs, dX_asp, type='asp', op='forth'):
        assert hasattr(dX_asp, 'redistribute'), dX_asp
        assert op in ['back', 'forth']
        return dX_asp

    def calculate_hamiltonian(self, wfs, kpt, dH_asp, H_MM, yy):
        avalues = self.get_a_values()

        dH_aii = dH_asp.partition.arraydict([setup.dO_ii.shape
                                             for setup in wfs.setups],
                                            dtype=wfs.dtype)

        for a in avalues:
            dH_aii[a][:] = yy * unpack(dH_asp[a][kpt.s])

        self.calculate(wfs, kpt.q, dH_aii, H_MM)

    def add_overlap_correction(self, wfs, S_qMM):
        avalues = self.get_a_values()
        dS_aii = [wfs.setups[a].dO_ii for a in avalues]
        dS_aii = dict(zip(avalues, dS_aii))  # XXX get rid of dict

        for a in dS_aii:
            dS_aii[a] = np.asarray(dS_aii[a], wfs.dtype)

        for q, S_MM in enumerate(S_qMM):
            self.calculate(wfs, q, dS_aii, S_MM)

    def gobble_data(self, wfs):
        pass  # Prepare internal data structures for calculate().

    def calculate(self, wfs, q, dX_aii, X_MM):
        raise NotImplementedError

    def get_a_values(self):
        raise NotImplementedError

    def implements_distributed_projections(self):
        return False

    def calculate_projections(self, wfs, kpt):
        for a, P_ni in kpt.P_ani.items():
            # ATLAS can't handle uninitialized output array:
            P_ni.fill(117)
            mmm(1.0, kpt.C_nM, 'N', wfs.P_aqMi[a][kpt.q], 'N', 0.0, P_ni)


class DenseAtomicCorrection(BaseAtomicCorrection):
    name = 'dense'
    description = 'dense with blas'

    def gobble_data(self, wfs):
        self.initialize(wfs.P_aqMi, wfs.ksl.Mstart, wfs.ksl.Mstop)
        self.orig_partition = wfs.atom_partition  # XXXXXXXXXXXXXXXXXXXXX

    def initialize(self, P_aqMi, Mstart, Mstop):
        self.P_aqMi = P_aqMi
        self.Mstart = Mstart
        self.Mstop = Mstop

    def get_a_values(self):
        return self.P_aqMi.keys()

    def calculate(self, wfs, q, dX_aii, X_MM):
        dtype = X_MM.dtype
        nops = 0

        # P_aqMi is distributed over domains (a) and bands (M).
        # Hence the correction X_MM = sum(P dX P) includes contributions
        # only from local atoms; the result must be summed over gd.comm
        # to get all 'a' contributions, and it will be locally calculated
        # only on the local slice of bands.

        for a, dX_ii in dX_aii.items():
            P_Mi = self.P_aqMi[a][q]
            assert dtype == P_Mi.dtype
            dXP_iM = np.zeros((dX_ii.shape[1], P_Mi.shape[0]), dtype)
            # (ATLAS can't handle uninitialized output array)
            gemm(1.0, P_Mi, dX_ii, 0.0, dXP_iM, 'c')
            nops += dXP_iM.size * dX_ii.shape[0]
            gemm(1.0, dXP_iM, P_Mi[self.Mstart:self.Mstop], 1.0, X_MM)
            nops += X_MM.size * dXP_iM.shape[0]
        self.nops = nops


class DistributedAtomicCorrection(BaseAtomicCorrection):
    name = 'distributed'
    description = 'distributed and block-sparse'

    def gobble_data(self, wfs):
        self.orig_partition = wfs.atom_partition
        evenpart = EvenPartitioning(self.orig_partition.comm,
                                    self.orig_partition.natoms)
        self.even_partition = evenpart.as_atom_partition()

    def get_a_values(self):
        return self.orig_partition.my_indices  # XXXXXXXXXX
        # return self.even_partition.my_indices

    def redistribute(self, wfs, dX_asp, type='asp', op='forth'):
        if type not in ['asp', 'aii']:
            raise ValueError('Unknown matrix type "%s"' % type)

        # just distributed over gd comm.  It's not the most aggressive
        # we can manage but we want to make band parallelization
        # a bit easier and it won't really be a problem.  I guess
        #
        # Also: This call is blocking, but we could easily do a
        # non-blocking version as we only need this stuff after
        # doing tons of real-space work.

        # XXXXXXXXXXXXXXXXXX
        if 1:
            return dX_asp.copy()

        dX_asp = dX_asp.copy()
        if op == 'forth':
            even = self.orig_partition.as_even_partition()
            dX_asp.redistribute(even)
        else:
            assert op == 'back'
            dX_asp.redistribute(self.orig_partition)
        return dX_asp

    def calculate(self, wfs, q, dX_aii, X_MM):
        # XXX reduce according to kpt.q
        dtype = wfs.dtype
        M_a = wfs.setups.M_a
        nM_a = np.array([setup.nao for setup in wfs.setups])
        Mstart = wfs.ksl.Mstart
        Mstop = wfs.ksl.Mstop

        # Now calculate basis-projector-basis overlap: a1 -> a3 -> a2
        #
        # specifically:
        #   < phi[a1] | p[a3] > * dX[a3] * < p[a3] | phi[a2] >
        #
        # This matrix multiplication is semi-sparse.  It works by blocks
        # of atoms, looping only over pairs that do have nonzero
        # overlaps.  But it might be even nicer with scipy sparse.
        # This we will have to check at some point.
        #
        # The projection arrays P_aaim are distributed over the grid,
        # whereas the X_MM is distributed over the band comm.
        # One could choose a set of a3 to optimize the load balance.
        # Right now the load balance will be "random" and probably
        # not very good.
        # innerloops = 0

        setups = wfs.setups

        outer = 0
        inner = 0
        nops = 0

        nao_a = [setup.nao for setup in setups]

        for (a3, a1), P1_qim in wfs.P_aaqim.items():
            P1_im = P1_qim[q]
            a1M1 = M_a[a1]
            nM1 = nM_a[a1]
            a1M2 = a1M1 + nM1

            if a1M1 > Mstop or a1M2 < Mstart:
                continue

            stickout1 = max(0, Mstart - a1M1)
            stickout2 = max(0, a1M2 - Mstop)
            P1_mi = np.conj(P1_im.T[stickout1:nM1 - stickout2])
            dX_ii = dX_aii[a3]
            X_mM = X_MM[a1M1 + stickout1 - Mstart:a1M2 - stickout2 - Mstart]

            P1dX_mi = np.dot(P1_mi, dX_ii)
            nops += P1dX_mi.size * dX_ii.shape[0]
            outer += 1

            assert len(wfs.P_neighbors_a[a3]) > 0
            if 0:
                for a2 in wfs.P_neighbors_a[a3]:
                    inner += 1
                    # We can use symmetry somehow.  Since the entire matrix
                    # is symmetrized after the X_MM is constructed,
                    # at least in the non-Gamma-point case, we should do
                    # so conditionally somehow.  Right now let's stay out
                    # of trouble.

                    # Humm.  The following works with gamma point
                    # but not with kpts.  XXX take a look at this.
                    # also, it doesn't work with a2 < a1 for some reason.
                    # if a2 > a1:
                    #     continue
                    a2M1 = M_a[a2]
                    a2M2 = a2M1 + wfs.setups[a2].nao
                    P2_im = wfs.P_aaqim[(a3, a2)][q]
                    P1dXP2_mm = np.dot(P1dX_mi, P2_im)
                    X_mM[:, a2M1:a2M2] += P1dXP2_mm
                    # innerloops += 1
                    # if wfs.world.rank == 0:
                    #     print 'y', y
                    # if a1 != a2:
                    #     X_MM[a2M1:a2M2, a1M1:a1M2] += P1dXP2_mm.T.conj()

            a2_a = wfs.P_neighbors_a[a3]

            a2nao = sum(nao_a[a2] for a2 in a2_a)

            P2_iam = np.empty((P1_mi.shape[1], a2nao), dtype)
            m2 = 0
            for a2 in a2_a:
                P2_im = wfs.P_aaqim[(a3, a2)][q]
                nao = nao_a[a2]
                P2_iam[:, m2:m2 + nao] = P2_im
                m2 += nao

            P1dXP2_mam = np.zeros((P1dX_mi.shape[0],
                                   P2_iam.shape[1]), dtype)
            gemm(1.0, P2_iam, P1dX_mi, 0.0, P1dXP2_mam)
            nops += P1dXP2_mam.size * P2_iam.shape[0]

            m2 = 0
            for a2 in a2_a:
                nao = nao_a[a2]
                X_mM[:, M_a[a2]:M_a[a2] + setups[a2].nao] += \
                    P1dXP2_mam[:, m2:m2 + nao]
                m2 += nao

        self.nops = nops
        self.inner = inner
        self.outer = outer


class ScipyAtomicCorrection(DistributedAtomicCorrection):
    name = 'scipy'
    description = 'distributed and sparse using scipy'

    def __init__(self, tolerance=1e-12):
        DistributedAtomicCorrection.__init__(self)
        from scipy import sparse
        self.sparse = sparse
        self.tolerance = tolerance

    def gobble_data(self, wfs):
        DistributedAtomicCorrection.gobble_data(self, wfs)
        nq = len(wfs.kd.ibzk_qc)

        I_a = [0]
        I_a.extend(np.cumsum([setup.ni for setup in wfs.setups[:-1]]))
        I = I_a[-1] + wfs.setups[-1].ni
        self.I = I
        self.I_a = I_a

        self.Psparse_qIM = wfs.P_qIM
        return

        M_a = wfs.setups.M_a
        M = M_a[-1] + wfs.setups[-1].nao
        nao_a = [setup.nao for setup in wfs.setups]
        ni_a = [setup.ni for setup in wfs.setups]

        Psparse_qIM = [self.sparse.lil_matrix((I, M), dtype=wfs.dtype)
                       for _ in range(nq)]

        for (a3, a1), P_qim in wfs.P_aaqim.items():
            if self.tolerance > 0:
                P_qim = P_qim.copy()
                approximately_zero = np.abs(P_qim) < self.tolerance
                P_qim[approximately_zero] = 0.0
            for q in range(nq):
                Psparse_qIM[q][I_a[a3]:I_a[a3] + ni_a[a3],
                               M_a[a1]:M_a[a1] + nao_a[a1]] = P_qim[q]

        self.Psparse_qIM = [x.tocsr() for x in Psparse_qIM]

    def calculate(self, wfs, q, dX_aii, X_MM):
        Mstart = wfs.ksl.Mstart
        Mstop = wfs.ksl.Mstop

        Psparse_IM = self.Psparse_qIM[q]

        dXsparse_II = self.sparse.lil_matrix((self.I, self.I), dtype=wfs.dtype)
        avalues = sorted(dX_aii.keys())
        for a in avalues:
            I1 = self.I_a[a]
            I2 = I1 + wfs.setups[a].ni
            dXsparse_II[I1:I2, I1:I2] = dX_aii[a]
        dXsparse_II = dXsparse_II.tocsr()

        Psparse_MI = Psparse_IM[:, Mstart:Mstop].transpose().conjugate()
        Xsparse_MM = Psparse_MI.dot(dXsparse_II.dot(Psparse_IM))
        X_MM[:, :] += Xsparse_MM.todense()

    def calculate_projections(self, wfs, kpt):
        if self.implements_distributed_projections():
            P_In = self.Psparse_qIM[kpt.q].dot(kpt.C_nM.T)
            # for a in self.even_partition.my_indices: # XXXXXXX
            for a in self.orig_partition.my_indices:
                I1 = self.I_a[a]
                I2 = I1 + wfs.setups[a].ni
                kpt.P_ani[a][:, :] = P_In[I1:I2, :].T.conj()
        else:
            DistributedAtomicCorrection.calculate_projections(self, wfs, kpt)

    def implements_distributed_projections(self):
        return True
