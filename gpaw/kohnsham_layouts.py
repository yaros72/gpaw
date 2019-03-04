# Copyright (C) 2010  CAMd
# Copyright (C) 2010  Argonne National Laboratory
# Please see the accompanying LICENSE file for further information.
import numpy as np

from gpaw.matrix_descriptor import MatrixDescriptor
from gpaw.mpi import broadcast_exception
from gpaw.blacs import BlacsGrid, Redistributor
from gpaw.utilities import uncamelcase
from gpaw.utilities.blas import gemm, r2k
from gpaw.utilities.lapack import general_diagonalize
from gpaw.utilities.scalapack import (pblas_simple_gemm, pblas_tran,
                                      scalapack_tri2full)
from gpaw.utilities.tools import tri2full
from gpaw.utilities.timing import nulltimer


def get_KohnSham_layouts(sl, mode, gd, bd, block_comm, dtype,
                         elpasolver=None, **kwargs):
    """Create Kohn-Sham layouts object."""
    # Not needed for AtomPAW special mode, as usual we just provide whatever
    # happens to make the code not crash
    if not isinstance(mode, str):
        return None  # XXX
    name = 'OrbitalLayouts'
    args = (gd, bd, block_comm, dtype)
    if sl is not None:
        name = 'Blacs' + name
        assert len(sl) == 3
        args += tuple(sl)
        if elpasolver is not None:
            kwargs['elpasolver'] = elpasolver
    ksl = {'BlacsOrbitalLayouts': BlacsOrbitalLayouts,
           'OrbitalLayouts': OrbitalLayouts}[name](*args,
                                                   **kwargs)
    return ksl


class KohnShamLayouts:
    using_blacs = False  # This is only used by a regression test
    matrix_descriptor_class = None
    accepts_decomposed_overlap_matrix = False

    def __init__(self, gd, bd, block_comm, dtype, timer=nulltimer):
        assert gd.comm.parent is bd.comm.parent  # must have same parent comm
        self.world = bd.comm.parent
        self.gd = gd
        self.bd = bd
        self.dtype = dtype
        self.block_comm = block_comm
        self.timer = timer
        self._kwargs = {'timer': timer}

        if gd.comm.rank == 0:
            self.column_comm = bd.comm
        else:
            self.column_comm = None

    def get_keywords(self):
        return self._kwargs.copy()  # just a shallow copy...

    def diagonalize(self, *args, **kwargs):
        raise RuntimeError('Virtual member function should not be called.')

    def inverse_cholesky(self, *args, **kwargs):
        raise RuntimeError('Virtual member function should not be called.')

    def new_descriptor(self):
        return self.matrix_descriptor_class(self.bd, self.gd, self)

    def __repr__(self):
        return uncamelcase(self.__class__.__name__)

    def get_description(self):
        """Description of this object in prose, e.g. for logging.

        Subclasses are expected to override this with something useful."""
        return repr(self)


class BlacsLayouts(KohnShamLayouts):
    using_blacs = True  # This is only used by a regression test

    def __init__(self, gd, bd, block_comm, dtype, mcpus, ncpus,
                 blocksize, timer=nulltimer):
        KohnShamLayouts.__init__(self, gd, bd, block_comm, dtype,
                                 timer)
        # WARNING: Do not create the BlacsGrid on a communicator which does not
        # contain block_comm.rank = 0. This will break BlacsBandLayouts which
        # assume eps_M will be broadcast over block_comm.
        self.blocksize = blocksize
        self.blockgrid = BlacsGrid(self.block_comm, mcpus, ncpus)

    def get_description(self):
        title = 'BLACS'
        template = '%d x %d grid with %d x %d blocksize'
        return (title, template)


class BlacsOrbitalLayouts(BlacsLayouts):
    """ScaLAPACK Dense Linear Algebra.

    This class is instantiated in LCAO.  Not for casual use, at least for now.

    Requires two distributors and three descriptors for initialization
    as well as grid descriptors and band descriptors. Distributors are
    for cols2blocks (1D -> 2D BLACS grid) and blocks2cols (2D -> 1D
    BLACS grid). ScaLAPACK operations must occur on 2D BLACS grid for
    performance and scalability.

    _general_diagonalize is "hard-coded" for LCAO.
    Expects both Hamiltonian and Overlap matrix to be on the 2D BLACS grid.
    This is done early on to save memory.
    """
    # XXX rewrite this docstring a bit!

    # This class 'describes' all the LCAO Blacs-related layouts
    def __init__(self, gd, bd, block_comm, dtype, mcpus, ncpus,
                 blocksize, nao, elpasolver=None, timer=nulltimer):
        BlacsLayouts.__init__(self, gd, bd, block_comm, dtype,
                              mcpus, ncpus, blocksize, timer)
        nbands = bd.nbands
        self.blocksize = blocksize

        self.orbital_comm = self.bd.comm
        self.naoblocksize = naoblocksize = -((-nao) // self.orbital_comm.size)
        self.nao = nao

        # Range of basis functions for BLACS distribution of matrices:
        self.Mmax = nao
        self.Mstart = min(bd.comm.rank * naoblocksize, self.Mmax)
        self.Mstop = min(self.Mstart + naoblocksize, self.Mmax)
        self.mynao = self.Mstop - self.Mstart

        # Column layout for one matrix per band rank:
        self.columngrid = BlacsGrid(bd.comm, bd.comm.size, 1)
        self.mMdescriptor = self.columngrid.new_descriptor(nao, nao,
                                                           naoblocksize, nao)
        self.nMdescriptor = self.columngrid.new_descriptor(nbands, nao,
                                                           bd.maxmynbands, nao)

        # parallelprint(world, (mynao, self.mMdescriptor.shape))

        # Column layout for one matrix in total (only on grid masters):
        self.single_column_grid = BlacsGrid(self.column_comm, bd.comm.size, 1)
        self.mM_unique_descriptor = self.single_column_grid.new_descriptor(
            nao, nao, naoblocksize, nao)

        # nM_unique_descriptor is meant to hold the coefficients after
        # diagonalization.  BLACS requires it to be nao-by-nao, but
        # we only fill meaningful data into the first nbands columns.
        #
        # The array will then be trimmed and broadcast across
        # the grid descriptor's communicator.
        self.nM_unique_descriptor = self.single_column_grid.new_descriptor(
            nbands, nao, bd.maxmynbands, nao)

        # Fully blocked grid for diagonalization with many CPUs:
        self.mmdescriptor = self.blockgrid.new_descriptor(nao, nao, blocksize,
                                                          blocksize)

        # self.nMdescriptor = nMdescriptor
        self.mM2mm = Redistributor(self.block_comm, self.mM_unique_descriptor,
                                   self.mmdescriptor)
        self.mm2nM = Redistributor(self.block_comm, self.mmdescriptor,
                                   self.nM_unique_descriptor)

        self.libelpa = None
        if elpasolver is not None:
            # XXX forward solver to libelpa
            from gpaw.utilities.elpa import LibElpa
            self.libelpa = LibElpa(self.mmdescriptor, solver=elpasolver,
                                   nev=nbands)

    @property
    def accepts_decomposed_overlap_matrix(self):
        return self.libelpa is not None

    def diagonalize(self, H_mm, C_nM, eps_n, S_mm,
                    is_already_decomposed=False):
        # C_nM needs to be simultaneously compatible with:
        # 1. outdescriptor
        # 2. broadcast with gd.comm
        # We will does this with a dummy buffer C2_nM
        outdescriptor = self.mm2nM.dstdescriptor  # blocks2cols
        blockdescriptor = self.mM2mm.dstdescriptor  # cols2blocks

        dtype = S_mm.dtype
        eps_M = np.empty(C_nM.shape[-1])  # empty helps us debug
        subM, subN = outdescriptor.gshape
        C_mm = blockdescriptor.zeros(dtype=dtype)

        self.timer.start('General diagonalize')
        # general_diagonalize_ex may have a buffer overflow, so
        # we no longer use it
        # blockdescriptor.general_diagonalize_ex(H_mm, S_mm.copy(), C_mm,
        #                                        eps_M,
        #                                        UL='L', iu=self.bd.nbands)
        if self.libelpa is not None:
            assert blockdescriptor is self.libelpa.desc
            scalapack_tri2full(blockdescriptor, H_mm)

            # elpa will write decomposed form of S_mm into S_mm.
            # Other KSL diagonalization functions do *not* overwrite S_mm.
            self.libelpa.general_diagonalize(
                H_mm, S_mm, C_mm, eps_M[:self.bd.nbands],
                is_already_decomposed=is_already_decomposed)
        else:
            blockdescriptor.general_diagonalize_dc(H_mm, S_mm.copy(), C_mm,
                                                   eps_M, UL='L')
        self.timer.stop('General diagonalize')

        # Make C_nM compatible with the redistributor
        self.timer.start('Redistribute coefs')
        if outdescriptor:
            C2_nM = C_nM
        else:
            C2_nM = outdescriptor.empty(dtype=dtype)
        assert outdescriptor.check(C2_nM)
        self.mm2nM.redistribute(C_mm, C2_nM, subM, subN)  # blocks2cols
        self.timer.stop('Redistribute coefs')

        self.timer.start('Send coefs to domains')
        # eps_M is already on block_comm.rank = 0
        # easier to broadcast eps_M to all and
        # get the correct slice afterward.
        self.block_comm.broadcast(eps_M, 0)
        eps_n[:] = eps_M[self.bd.get_slice()]
        self.gd.comm.broadcast(C_nM, 0)
        self.timer.stop('Send coefs to domains')

    def distribute_overlap_matrix(self, S_qmM, root=0,
                                  add_hermitian_conjugate=False):
        # Some MPI implementations need a lot of memory to do large
        # reductions.  To avoid trouble, we do comm.sum on smaller blocks
        # of S (this code is also safe for arrays smaller than blocksize)
        Sflat_x = S_qmM.ravel()
        blocksize = 2**23 // Sflat_x.itemsize  # 8 MiB
        nblocks = -(-len(Sflat_x) // blocksize)
        Mstart = 0
        self.timer.start('blocked summation')
        for i in range(nblocks):
            self.gd.comm.sum(Sflat_x[Mstart:Mstart + blocksize], root=root)
            Mstart += blocksize
        assert Mstart + blocksize >= len(Sflat_x)
        self.timer.stop('blocked summation')

        xshape = S_qmM.shape[:-2]
        if len(xshape) == 0:
            S_qmM = S_qmM[np.newaxis]
        assert S_qmM.ndim == 3

        blockdesc = self.mmdescriptor
        coldesc = self.mM_unique_descriptor
        S_qmm = blockdesc.zeros(len(S_qmM), S_qmM.dtype)

        if not coldesc:  # XXX ugly way to sort out inactive ranks
            S_qmM = coldesc.zeros(len(S_qmM), S_qmM.dtype)

        self.timer.start('Scalapack redistribute')
        for S_mM, S_mm in zip(S_qmM, S_qmm):
            self.mM2mm.redistribute(S_mM, S_mm)
            if add_hermitian_conjugate:
                if blockdesc.active:
                    pblas_tran(1.0, S_mm.copy(), 1.0, S_mm,
                               blockdesc, blockdesc)

            if self.libelpa is not None:
                # Elpa needs the full H_mm, but apparently does not
                # need the full S_mm.  However that fact is not documented,
                # and hence we stay on the safe side, tri2full-ing the
                # matrix.
                scalapack_tri2full(blockdesc, S_mm)

        self.timer.stop('Scalapack redistribute')
        return S_qmm.reshape(xshape + blockdesc.shape)

    def get_overlap_matrix_shape(self):
        return self.mmdescriptor.shape

    def calculate_blocked_density_matrix(self, f_n, C_nM):
        nbands = self.bd.nbands
        nao = self.nao
        dtype = C_nM.dtype

        self.nMdescriptor.checkassert(C_nM)
        if self.gd.rank == 0:
            Cf_nM = (C_nM * f_n[:, None])
        else:
            C_nM = self.nM_unique_descriptor.zeros(dtype=dtype)
            Cf_nM = self.nM_unique_descriptor.zeros(dtype=dtype)

        r = Redistributor(self.block_comm, self.nM_unique_descriptor,
                          self.mmdescriptor)

        Cf_mm = self.mmdescriptor.zeros(dtype=dtype)
        r.redistribute(Cf_nM, Cf_mm, nbands, nao)
        del Cf_nM

        C_mm = self.mmdescriptor.zeros(dtype=dtype)
        r.redistribute(C_nM, C_mm, nbands, nao)
        # no use to delete C_nM as it's in the input...

        rho_mm = self.mmdescriptor.zeros(dtype=dtype)

        if 1:  # if self.libelpa is None:
            pblas_simple_gemm(self.mmdescriptor,
                              self.mmdescriptor,
                              self.mmdescriptor,
                              Cf_mm, C_mm, rho_mm, transa='C')
        else:
            # elpa_hermitian_multiply was not faster than the ordinary
            # multiplication in the test.  The way we have things distributed,
            # we need to transpose things at the moment.
            #
            # Rather than enabling this, we should store the coefficients
            # in an appropriate 2D block cyclic format (c_nm) and not the
            # current C_nM format.  This makes it possible to avoid
            # redistributing the coefficients at all.  But we don't have time
            # to implement this at the moment.
            mul = self.libelpa.hermitian_multiply
            desc = self.mmdescriptor
            from gpaw.utilities.scalapack import pblas_tran

            def T(array):
                tmp = array.copy()
                pblas_tran(alpha=1.0, a_MN=tmp,
                           beta=0.0, c_NM=array,
                           desca=desc, descc=desc)
            T(C_mm)
            T(Cf_mm)
            mul(C_mm, Cf_mm, rho_mm,
                desc, desc, desc,
                uplo_a='X', uplo_c='X')

        return rho_mm

    def calculate_density_matrix(self, f_n, C_nM, rho_mM=None):
        """Calculate density matrix from occupations and coefficients.

        Presently this function performs the usual scalapack 3-step trick:
        redistribute-numbercrunching-backdistribute.


        Notes on future performance improvement.

        As per the current framework, C_nM exists as copies on each
        domain, i.e. this is not parallel over domains.  We'd like to
        correct this and have an efficient distribution using e.g. the
        block communicator.

        The diagonalization routine and other parts of the code should
        however be changed to accommodate the following scheme:

        Keep coefficients in C_mm form after the diagonalization.
        rho_mm can then be directly calculated from C_mm without
        redistribution, after which we only need to redistribute
        rho_mm across domains.

        """
        dtype = C_nM.dtype
        rho_mm = self.calculate_blocked_density_matrix(f_n, C_nM)
        rback = Redistributor(self.block_comm, self.mmdescriptor,
                              self.mM_unique_descriptor)
        rho1_mM = self.mM_unique_descriptor.zeros(dtype=dtype)
        rback.redistribute(rho_mm, rho1_mM)
        del rho_mm

        if rho_mM is None:
            if self.gd.rank == 0:
                rho_mM = rho1_mM
            else:
                rho_mM = self.mMdescriptor.zeros(dtype=dtype)

        self.gd.comm.broadcast(rho_mM, 0)
        return rho_mM

    def distribute_to_columns(self, rho_mm, srcdescriptor):
        redistributor = Redistributor(self.block_comm,  # XXX
                                      srcdescriptor,
                                      self.mM_unique_descriptor)
        rho_mM = redistributor.redistribute(rho_mm)
        if self.gd.rank != 0:
            rho_mM = self.mMdescriptor.zeros(dtype=rho_mm.dtype)
        self.gd.comm.broadcast(rho_mM, 0)
        return rho_mM

    def oldcalculate_density_matrix(self, f_n, C_nM, rho_mM=None):
        # This version is parallel over the band descriptor only.
        # This is inefficient, but let's keep it for a while in case
        # there's trouble with the more efficient version
        if rho_mM is None:
            rho_mM = self.mMdescriptor.zeros(dtype=C_nM.dtype)

        Cf_nM = (C_nM * f_n[:, None]).conj()
        pblas_simple_gemm(self.nMdescriptor, self.nMdescriptor,
                          self.mMdescriptor, Cf_nM, C_nM, rho_mM, transa='T')
        return rho_mM

    def get_transposed_density_matrix(self, f_n, C_nM, rho_mM=None):
        return self.calculate_density_matrix(f_n, C_nM, rho_mM).conj()

    def get_description(self):
        (title, template) = BlacsLayouts.get_description(self)
        bg = self.blockgrid
        desc = self.mmdescriptor
        s = template % (bg.nprow, bg.npcol, desc.mb, desc.nb)
        if self.libelpa is not None:
            solver = self.libelpa.description
        else:
            solver = 'ScaLAPACK'
        return ''.join([title, ' / ', solver, ', ', s])


class OrbitalLayouts(KohnShamLayouts):
    def __init__(self, gd, bd, block_comm, dtype, nao,
                 timer=nulltimer):
        KohnShamLayouts.__init__(self, gd, bd, block_comm, dtype,
                                 timer)
        self.mMdescriptor = MatrixDescriptor(nao, nao)
        self.nMdescriptor = MatrixDescriptor(bd.mynbands, nao)

        self.Mstart = 0
        self.Mstop = nao
        self.Mmax = nao
        self.mynao = nao
        self.nao = nao
        self.orbital_comm = bd.comm

    def diagonalize(self, H_MM, C_nM, eps_n, S_MM,
                    overwrite_S=False):
        assert not overwrite_S
        eps_M = np.empty(C_nM.shape[-1])
        self.block_comm.broadcast(H_MM, 0)
        self.block_comm.broadcast(S_MM, 0)
        # The result on different processor is not necessarily bit-wise
        # identical, so only domain master performs computation
        with broadcast_exception(self.gd.comm):
            if self.gd.comm.rank == 0:
                self._diagonalize(H_MM, S_MM.copy(), eps_M)
        self.gd.comm.broadcast(H_MM, 0)
        self.gd.comm.broadcast(eps_M, 0)
        eps_n[:] = eps_M[self.bd.get_slice()]
        C_nM[:] = H_MM[self.bd.get_slice()]

    def _diagonalize(self, H_MM, S_MM, eps_M):
        """Serial diagonalize via LAPACK."""
        # This is replicated computation but ultimately avoids
        # additional communication
        general_diagonalize(H_MM, eps_M, S_MM)

    def estimate_memory(self, mem, dtype):
        itemsize = mem.itemsize[dtype]
        mem.subnode('eps [M]', self.nao * mem.floatsize)
        mem.subnode('H [MM]', self.nao * self.nao * itemsize)

    def distribute_overlap_matrix(self, S_qMM, root=0,
                                  add_hermitian_conjugate=False):
        self.gd.comm.sum(S_qMM, root)
        if add_hermitian_conjugate:
            S_qMM += S_qMM.swapaxes(-1, -2).conj()
        return S_qMM

    def get_overlap_matrix_shape(self):
        return self.nao, self.nao

    def calculate_density_matrix(self, f_n, C_nM, rho_MM=None, C2_nM=None):
        # Only a madman would use a non-transposed density matrix.
        # Maybe we should use the get_transposed_density_matrix instead
        if rho_MM is None:
            rho_MM = np.zeros((self.mynao, self.nao), dtype=C_nM.dtype)
        # XXX Should not conjugate, but call gemm(..., 'c')
        # Although that requires knowing C_Mn and not C_nM.
        # that also conforms better to the usual conventions in literature
        if C2_nM is None:
            C2_nM = C_nM
        Cf_Mn = np.ascontiguousarray(C2_nM.T.conj() * f_n)
        gemm(1.0, C_nM, Cf_Mn, 0.0, rho_MM, 'n')
        return rho_MM

    def get_transposed_density_matrix(self, f_n, C_nM, rho_MM=None):
        return self.calculate_density_matrix(f_n, C_nM, rho_MM).T.copy()

        # if rho_MM is None:
        #     rho_MM = np.zeros((self.mynao, self.nao), dtype=C_nM.dtype)
        # C_Mn = C_nM.T.copy()
        # gemm(1.0, C_Mn, f_n[np.newaxis, :] * C_Mn, 0.0, rho_MM, 'c')
        # self.bd.comm.sum(rho_MM)
        # return rho_MM

    def alternative_calculate_density_matrix(self, f_n, C_nM, rho_MM=None):
        if rho_MM is None:
            rho_MM = np.zeros((self.mynao, self.nao), dtype=C_nM.dtype)
        # Alternative suggestion. Might be faster. Someone should test this
        C_Mn = C_nM.T.copy()
        r2k(0.5, C_Mn, f_n * C_Mn, 0.0, rho_MM)
        tri2full(rho_MM)
        return rho_MM

    def get_description(self):
        return 'Serial LAPACK'

    def calculate_density_matrix_delta(self, d_nn, C_nM, rho_MM=None):
        # Only a madman would use a non-transposed density matrix.
        # Maybe we should use the get_transposed_density_matrix instead
        if rho_MM is None:
            rho_MM = np.zeros((self.mynao, self.nao), dtype=C_nM.dtype)
        Cd_Mn = np.zeros((self.nao, self.bd.mynbands), dtype=C_nM.dtype)
        # XXX Should not conjugate, but call gemm(..., 'c')
        # Although that requires knowing C_Mn and not C_nM.
        # that also conforms better to the usual conventions in literature
        C_Mn = C_nM.T.conj().copy()
        gemm(1.0, d_nn, C_Mn, 0.0, Cd_Mn, 'n')
        gemm(1.0, C_nM, Cd_Mn, 0.0, rho_MM, 'n')
        self.bd.comm.sum(rho_MM)
        return rho_MM

    def get_transposed_density_matrix_delta(self, d_nn, C_nM, rho_MM=None):
        return self.calculate_density_matrix_delta(d_nn, C_nM, rho_MM).T.copy()
