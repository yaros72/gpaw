"""Module defining an eigensolver base-class."""
from functools import partial

import numpy as np
from ase.utils.timing import timer

from gpaw.matrix import matrix_matrix_multiply as mmm
from gpaw.utilities.blas import axpy
from gpaw.xc.hybrid import HybridXC


def reshape(a_x, shape):
    """Get an ndarray of size shape from a_x buffer."""
    return a_x.ravel()[:np.prod(shape)].reshape(shape)


class Eigensolver:
    def __init__(self, keep_htpsit=True, blocksize=1):
        self.keep_htpsit = keep_htpsit
        self.initialized = False
        self.Htpsit_nG = None
        self.error = np.inf
        self.blocksize = blocksize
        self.orthonormalization_required = True

    def initialize(self, wfs):
        self.timer = wfs.timer
        self.world = wfs.world
        self.kpt_comm = wfs.kd.comm
        self.band_comm = wfs.bd.comm
        self.dtype = wfs.dtype
        self.bd = wfs.bd
        self.nbands = wfs.bd.nbands
        self.mynbands = wfs.bd.mynbands

        if wfs.bd.comm.size > 1:
            self.keep_htpsit = False

        if self.keep_htpsit:
            self.Htpsit_nG = np.empty_like(wfs.work_array)

        # Preconditioner for the electronic gradients:
        self.preconditioner = wfs.make_preconditioner(self.blocksize)

        for kpt in wfs.kpt_u:
            if kpt.eps_n is None:
                kpt.eps_n = np.empty(self.mynbands)

        self.initialized = True

    def reset(self):
        self.initialized = False

    def weights(self, kpt):
        if self.nbands_converge == 'occupied':
            if kpt.f_n is None:
                weights = np.empty(self.bd.mynbands)
                weights[:] = kpt.weight
            else:
                weights = kpt.f_n
        else:
            weights = np.zeros(self.bd.mynbands)
            n = self.nbands_converge - self.bd.beg
            if n > 0:
                weights[:n] = kpt.weight
        return weights

    def iterate(self, ham, wfs):
        """Solves eigenvalue problem iteratively

        This method is inherited by the actual eigensolver which should
        implement *iterate_one_k_point* method for a single iteration of
        a single kpoint.
        """

        if not self.initialized:
            if isinstance(ham.xc, HybridXC):
                self.blocksize = wfs.bd.mynbands
            self.initialize(wfs)

        error = 0.0
        for kpt in wfs.kpt_u:
            if not wfs.orthonormalized:
                wfs.orthonormalize(kpt)
            e = self.iterate_one_k_point(ham, wfs, kpt)
            error += e
            if self.orthonormalization_required:
                wfs.orthonormalize(kpt)

        wfs.orthonormalized = True
        self.error = self.band_comm.sum(self.kpt_comm.sum(error))

    def iterate_one_k_point(self, ham, kpt):
        """Implemented in subclasses."""
        raise NotImplementedError

    def calculate_residuals(self, kpt, wfs, ham, psit, P, eps_n,
                            R, C, n_x=None, calculate_change=False):
        """Calculate residual.

        From R=Ht*psit calculate R=H*psit-eps*S*psit."""

        for R_G, eps, psit_G in zip(R.array, eps_n, psit.array):
            axpy(-eps, psit_G, R_G)

        ham.dH(P, out=C)
        for a, I1, I2 in P.indices:
            dS_ii = ham.setups[a].dO_ii
            C.array[..., I1:I2] -= np.dot((P.array[..., I1:I2].T * eps_n).T,
                                          dS_ii)

        ham.xc.add_correction(kpt, psit.array, R.array,
                              {a: P_ni for a, P_ni in P.items()},
                              {a: C_ni for a, C_ni in C.items()},
                              n_x,
                              calculate_change)
        wfs.pt.add(R.array, {a: C_ni for a, C_ni in C.items()}, kpt.q)

    @timer('Subspace diag')
    def subspace_diagonalize(self, ham, wfs, kpt):
        """Diagonalize the Hamiltonian in the subspace of kpt.psit_nG

        *Htpsit_nG* is a work array of same size as psit_nG which contains
        the local part of the Hamiltonian times psit on exit

        First, the Hamiltonian (defined by *kin*, *vt_sG*, and
        *dH_asp*) is applied to the wave functions, then the *H_nn*
        matrix is calculated and diagonalized, and finally, the wave
        functions (and also Htpsit_nG are rotated.  Also the
        projections *P_ani* are rotated.

        It is assumed that the wave functions *psit_nG* are orthonormal
        and that the integrals of projector functions and wave functions
        *P_ani* are already calculated.

        Return rotated wave functions and H applied to the rotated
        wave functions if self.keep_htpsit is True.
        """

        if self.band_comm.size > 1 and wfs.bd.strided:
            raise NotImplementedError

        psit = kpt.psit
        tmp = psit.new(buf=wfs.work_array)
        H = wfs.work_matrix_nn
        P2 = kpt.projections.new()

        Ht = partial(wfs.apply_pseudo_hamiltonian, kpt, ham)

        with self.timer('calc_h_matrix'):
            # We calculate the complex conjugate of H, because
            # that is what is most efficient with BLAS given the layout of
            # our matrices.
            psit.matrix_elements(operator=Ht, result=tmp, out=H,
                                 symmetric=True, cc=True)
            ham.dH(kpt.projections, out=P2)
            mmm(1.0, kpt.projections, 'N', P2, 'C', 1.0, H, symmetric=True)
            ham.xc.correct_hamiltonian_matrix(kpt, H.array)

        with wfs.timer('diagonalize'):
            slcomm, r, c, b = wfs.scalapack_parameters
            if r == c == 1:
                slcomm = None
            else:
                ranks = [rbd * wfs.gd.comm.size + rgd
                         for rgd in range(wfs.gd.comm.size)
                         for rbd in range(wfs.bd.comm.size)]
                slcomm = slcomm.new_communicator(ranks)
            # Complex conjugate before diagonalizing:
            eps_n = H.eigh(cc=True, scalapack=(slcomm, r, c, b))
            # H.array[n, :] now contains the n'th eigenvector and eps_n[n]
            # the n'th eigenvalue
            kpt.eps_n = eps_n[wfs.bd.get_slice()]

        with self.timer('rotate_psi'):
            if self.keep_htpsit:
                Htpsit = psit.new(buf=self.Htpsit_nG)
                mmm(1.0, H, 'N', tmp, 'N', 0.0, Htpsit)
            mmm(1.0, H, 'N', psit, 'N', 0.0, tmp)
            psit[:] = tmp
            mmm(1.0, H, 'N', kpt.projections, 'N', 0.0, P2)
            kpt.projections.matrix = P2.matrix
            # Rotate orbital dependent XC stuff:
            ham.xc.rotate(kpt, H.array.T)

    def estimate_memory(self, mem, wfs):
        gridmem = wfs.bytes_per_wave_function()

        keep_htpsit = self.keep_htpsit and (wfs.bd.mynbands == wfs.bd.nbands)

        if keep_htpsit:
            mem.subnode('Htpsit', wfs.bd.nbands * gridmem)
        else:
            mem.subnode('No Htpsit', 0)

        mem.subnode('eps_n', wfs.bd.mynbands * mem.floatsize)
        mem.subnode('eps_N', wfs.bd.nbands * mem.floatsize)
        mem.subnode('Preconditioner', 4 * gridmem)
        mem.subnode('Work', gridmem)
