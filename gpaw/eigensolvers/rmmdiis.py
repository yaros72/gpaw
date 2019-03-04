"""Module defining  ``Eigensolver`` classes."""
from functools import partial

import numpy as np

from gpaw.utilities.blas import axpy
from gpaw.eigensolvers.eigensolver import Eigensolver


class RMMDIIS(Eigensolver):
    """RMM-DIIS eigensolver

    It is expected that the trial wave functions are orthonormal
    and the integrals of projector functions and wave functions
    ``nucleus.P_uni`` are already calculated

    Solution steps are:

    * Subspace diagonalization
    * Calculation of residuals
    * Improvement of wave functions:  psi' = psi + lambda PR + lambda PR'
    * Orthonormalization"""

    def __init__(self, keep_htpsit=True, blocksize=None, niter=3, rtol=1e-16,
                 limit_lambda=False, use_rayleigh=False, trial_step=0.1):
        """Initialize RMM-DIIS eigensolver.

        Parameters:

        limit_lambda: dictionary
            determines if step length should be limited
            supported keys:
            'absolute':True/False limit the absolute value
            'upper':float upper limit for lambda
            'lower':float lower limit for lambda

        """

        Eigensolver.__init__(self, keep_htpsit, blocksize)
        self.niter = niter
        self.rtol = rtol
        self.limit_lambda = limit_lambda
        self.use_rayleigh = use_rayleigh
        if use_rayleigh:
            1 / 0
            self.blocksize = 1
        self.trial_step = trial_step
        self.first = True

    def todict(self):
        return {'name': 'rmm-diis', 'niter': self.niter}

    def initialize(self, wfs):
        if self.blocksize is None:
            if wfs.mode == 'pw':
                S = wfs.pd.comm.size
                # Use a multiple of S for maximum efficiency
                self.blocksize = int(np.ceil(10 / S)) * S
            else:
                self.blocksize = 10
        Eigensolver.initialize(self, wfs)

    def iterate_one_k_point(self, ham, wfs, kpt):
        """Do a single RMM-DIIS iteration for the kpoint"""

        self.subspace_diagonalize(ham, wfs, kpt)

        psit = kpt.psit
        # psit2 = psit.new(buf=wfs.work_array)
        P = kpt.projections
        P2 = P.new()
        # dMP = P.new()
        # M_nn = wfs.work_matrix_nn
        # dS = wfs.setups.dS
        R = psit.new(buf=self.Htpsit_nG)

        self.timer.start('RMM-DIIS')
        if self.keep_htpsit:
            with self.timer('Calculate residuals'):
                self.calculate_residuals(kpt, wfs, ham, psit, P, kpt.eps_n,
                                         R, P2)

        def integrate(a_G, b_G):
            return np.real(wfs.integrate(a_G, b_G, global_integral=False))

        comm = wfs.gd.comm

        B = self.blocksize
        dR = R.new(dist=None, nbands=B)
        dpsit = dR.new()
        P = P.new(bcomm=None, nbands=B)
        P2 = P.new()
        errors_x = np.zeros(B)

        # Arrays needed for DIIS step
        if self.niter > 1:
            psit_diis_nxG = wfs.empty(B * self.niter, q=kpt.q)
            R_diis_nxG = wfs.empty(B * self.niter, q=kpt.q)

        weights = self.weights(kpt)

        Ht = partial(wfs.apply_pseudo_hamiltonian, kpt, ham)

        error = 0.0
        for n1 in range(0, wfs.bd.mynbands, B):
            n2 = n1 + B
            if n2 > wfs.bd.mynbands:
                n2 = wfs.bd.mynbands
                B = n2 - n1
                P = P.new(nbands=B)
                P2 = P.new()
                dR = dR.new(nbands=B, dist=None)
                dpsit = dR.new()

            n_x = np.arange(n1, n2)
            psitb = psit.view(n1, n2)

            with self.timer('Calculate residuals'):
                Rb = R.view(n1, n2)
                if not self.keep_htpsit:
                    psitb.apply(Ht, out=Rb)
                    psitb.matrix_elements(wfs.pt, out=P)
                    self.calculate_residuals(kpt, wfs, ham, psitb,
                                             P, kpt.eps_n[n_x], Rb, P2, n_x)

            errors_x[:] = 0.0
            for n in range(n1, n2):
                weight = weights[n]
                errors_x[n - n1] = weight * integrate(Rb.array[n - n1],
                                                      Rb.array[n - n1])
            comm.sum(errors_x)
            error += np.sum(errors_x)

            # Insert first vectors and residuals for DIIS step
            if self.niter > 1:
                # Save the previous vectors contiguously for each band
                # in the block
                psit_diis_nxG[:B * self.niter:self.niter] = psitb.array
                R_diis_nxG[:B * self.niter:self.niter] = Rb.array

            # Precondition the residual:
            with self.timer('precondition'):
                ekin_x = self.preconditioner.calculate_kinetic_energy(
                    psitb.array, kpt)
                self.preconditioner(Rb.array, kpt, ekin_x, out=dpsit.array)

            # Calculate the residual of dpsit_G, dR_G = (H - e S) dpsit_G:
            # self.timer.start('Apply Hamiltonian')
            dpsit.apply(Ht, out=dR)
            # self.timer.stop('Apply Hamiltonian')
            with self.timer('projections'):
                dpsit.matrix_elements(wfs.pt, out=P)

            with self.timer('Calculate residuals'):
                self.calculate_residuals(kpt, wfs, ham, dpsit,
                                         P, kpt.eps_n[n_x], dR, P2, n_x,
                                         calculate_change=True)

            # Find lam that minimizes the norm of R'_G = R_G + lam dR_G
            with self.timer('Find lambda'):
                RdR_x = np.array([integrate(dR_G, R_G)
                                  for R_G, dR_G in zip(Rb.array, dR.array)])
                dRdR_x = np.array([integrate(dR_G, dR_G) for dR_G in dR.array])
                comm.sum(RdR_x)
                comm.sum(dRdR_x)
                lam_x = -RdR_x / dRdR_x

            # Limit abs(lam) to [0.15, 1.0]
            if self.limit_lambda:
                upper = self.limit_lambda['upper']
                lower = self.limit_lambda['lower']
                if self.limit_lambda.get('absolute', False):
                    lam_x = np.where(np.abs(lam_x) < lower,
                                     lower * np.sign(lam_x), lam_x)
                    lam_x = np.where(np.abs(lam_x) > upper,
                                     upper * np.sign(lam_x), lam_x)
                else:
                    lam_x = np.where(lam_x < lower, lower, lam_x)
                    lam_x = np.where(lam_x > upper, upper, lam_x)

            # lam_x[:] = 0.1

            # New trial wavefunction and residual
            with self.timer('Update psi'):
                for lam, psit_G, dpsit_G, R_G, dR_G in zip(
                        lam_x, psitb.array,
                        dpsit.array, Rb.array,
                        dR.array):
                    axpy(lam, dpsit_G, psit_G)  # psit_G += lam * dpsit_G
                    axpy(lam, dR_G, R_G)  # R_G += lam** dR_G

            self.timer.start('DIIS step')
            # DIIS step
            for nit in range(1, self.niter):
                # Do not perform DIIS if error is small
                # if abs(error_block / B) < self.rtol:
                #     break

                # Update the subspace
                psit_diis_nxG[nit:B * self.niter:self.niter] = psitb.array
                R_diis_nxG[nit:B * self.niter:self.niter] = Rb.array

                # XXX Only integrals of nit old psits would be needed
                # self.timer.start('projections')
                # wfs.pt.integrate(psit_diis_nxG, P_diis_anxi, kpt.q)
                # self.timer.stop('projections')
                if nit > 1 or self.limit_lambda:
                    for ib in range(B):
                        istart = ib * self.niter
                        iend = istart + nit + 1

                        # Residual matrix
                        self.timer.start('Construct matrix')
                        R_nn = wfs.integrate(R_diis_nxG[istart:iend],
                                             R_diis_nxG[istart:iend],
                                             global_integral=True)

                        # Full matrix
                        A_nn = -np.ones((nit + 2, nit + 2), wfs.dtype)
                        A_nn[:nit + 1, :nit + 1] = R_nn[:]
                        A_nn[-1, -1] = 0.0
                        x_n = np.zeros(nit + 2, wfs.dtype)
                        x_n[-1] = -1.0
                        self.timer.stop('Construct matrix')
                        with self.timer('Linear solve'):
                            alpha_i = np.linalg.solve(A_nn, x_n)[:-1]

                        self.timer.start('Update trial vectors')
                        psitb.array[ib] = alpha_i[nit] * psit_diis_nxG[istart +
                                                                       nit]
                        Rb.array[ib] = alpha_i[nit] * R_diis_nxG[istart + nit]
                        for i in range(nit):
                            # axpy(alpha_i[i], psit_diis_nxG[istart + i],
                            #      psit_diis_nxG[istart + nit])
                            # axpy(alpha_i[i], R_diis_nxG[istart + i],
                            #      R_diis_nxG[istart + nit])
                            axpy(alpha_i[i], psit_diis_nxG[istart + i],
                                 psitb.array[ib])
                            axpy(alpha_i[i], R_diis_nxG[istart + i],
                                 Rb.array[ib])
                        self.timer.stop('Update trial vectors')

                if nit < self.niter - 1:
                    with self.timer('precondition'):
                        self.preconditioner(Rb.array, kpt,
                                            ekin_x, out=dpsit.array)

                    for psit_G, lam, dpsit_G in zip(psitb.array, lam_x,
                                                    dpsit.array):
                        axpy(lam, dpsit_G, psit_G)

                    # Calculate the new residuals
                    self.timer.start('Calculate residuals')
                    psitb.apply(Ht, out=Rb)
                    psitb.matrix_elements(wfs.pt, out=P)
                    self.calculate_residuals(kpt, wfs, ham, psitb,
                                             P, kpt.eps_n[n_x], Rb, P2, n_x,
                                             calculate_change=True)
                    self.timer.stop('Calculate residuals')

            self.timer.stop('DIIS step')
            # Final trial step
            with self.timer('precondition'):
                self.preconditioner(Rb.array, kpt, ekin_x, out=dpsit.array)

            self.timer.start('Update psi')
            if self.trial_step is not None:
                lam_x[:] = self.trial_step
            for lam, psit_G, dpsit_G in zip(lam_x, psitb.array, dpsit.array):
                axpy(lam, dpsit_G, psit_G)  # psit_G += lam * dpsit_G
            self.timer.stop('Update psi')

        self.timer.stop('RMM-DIIS')
        return error

    def __repr__(self):
        repr_string = 'RMM-DIIS eigensolver\n'
        repr_string += '       keep_htpsit: %s\n' % self.keep_htpsit
        repr_string += '       DIIS iterations: %d\n' % self.niter
        repr_string += '       Threshold for DIIS: %5.1e\n' % self.rtol
        repr_string += '       Limit lambda: %s\n' % self.limit_lambda
        repr_string += '       use_rayleigh: %s\n' % self.use_rayleigh
        repr_string += '       trial_step: %s' % self.trial_step
        return repr_string
