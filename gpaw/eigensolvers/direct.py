"""Module defining  ``Eigensolver`` classes."""


import numpy as np
from gpaw.eigensolvers.eigensolver import Eigensolver
from gpaw.matrix_descriptor import MatrixDescriptor


class DirectPW(Eigensolver):
    """Direct eigensolver for plane waves"""

    def __init__(self, keep_htpsit=False):
        """Initialize direct eigensolver. """

        Eigensolver.__init__(self, keep_htpsit)

    def iterate_one_k_point(self, ham, wfs, kpt):
        """Setup H and S matries and diagonalize for the kpoint"""

        self.timer.start('DirectPW')
        H_GG, S_GG = wfs.hs(ham, kpt.q, kpt.s)
        npw = len(H_GG)
        eps_n = np.empty(npw)
        md = MatrixDescriptor(npw, npw)
        psit_nG = md.empty(dtype=complex)
        md.general_diagonalize_dc(H_GG, S_GG, psit_nG, eps_n,
                                  iu=wfs.bd.nbands)
        kpt.eps_n[:] = eps_n[:wfs.bd.nbands]
        wfs.pt.integrate(kpt.psit_nG, kpt.P_ani, kpt.q)
        self.timer.stop('DirectPW')
        error = 0.0
        return error

    def __repr__(self):
        return 'DirectPW eigensolver'
