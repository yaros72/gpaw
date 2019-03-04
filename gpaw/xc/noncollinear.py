import numpy as np

from gpaw.lcao.eigensolver import DirectLCAO


class NonCollinearLDAKernel:
    name = 'LDA'
    type = 'LDA'

    def __init__(self, kernel):
        self.kernel = kernel

    def calculate(self, e_g, n_sg, v_sg):
        n_g = n_sg[0]
        m_vg = n_sg[1:4]
        m_g = (m_vg**2).sum(0)**0.5
        nnew_sg = np.empty((2,) + n_g.shape)
        nnew_sg[:] = n_g
        nnew_sg[0] += m_g
        nnew_sg[1] -= m_g
        nnew_sg *= 0.5
        vnew_sg = np.zeros_like(nnew_sg)
        self.kernel.calculate(e_g, nnew_sg, vnew_sg)
        v_sg[0] += 0.5 * vnew_sg.sum(0)
        vnew_sg /= np.where(m_g < 1e-15, 1, m_g)
        v_sg[1:4] += 0.5 * vnew_sg[0] * m_vg
        v_sg[1:4] -= 0.5 * vnew_sg[1] * m_vg


class NonCollinearLCAOEigensolver(DirectLCAO):
    def iterate(self, ham, wfs):
        wfs.timer.start('LCAO eigensolver')

        wfs.timer.start('Potential matrix')
        Vt_xdMM = [wfs.basis_functions.calculate_potential_matrices(vt_G)
                   for vt_G in ham.vt_xG]
        wfs.timer.stop('Potential matrix')

        for kpt in wfs.mykpts:
            self.iterate_one_k_point(ham, wfs, kpt, Vt_xdMM)

        wfs.timer.stop('LCAO eigensolver')

    def iterate_one_k_point(self, ham, wfs, kpt, Vt_xdMM):
        assert wfs.gd.comm.size == 1, 'No quite sure this works!'
        if wfs.bd.comm.size > 1 and wfs.bd.strided:
            raise NotImplementedError

        H_xMM = []
        for x in range(4):
            kpt.s = x
            H_MM = self.calculate_hamiltonian_matrix(ham, wfs, kpt, Vt_xdMM[x],
                                                     root=0,
                                                     add_kinetic=(x == 0))
            H_xMM.append(H_MM)
        kpt.s = None

        S_MM = wfs.S_qMM[kpt.q]
        M = len(S_MM)
        S2_MM = np.zeros((2 * M, 2 * M), complex)
        H2_MM = np.zeros((2 * M, 2 * M), complex)

        S2_MM[:M, :M] = S_MM
        S2_MM[M:, M:] = S_MM

        H2_MM[:M, :M] = H_xMM[0] + H_xMM[3]
        H2_MM[M:, M:] = H_xMM[0] - H_xMM[3]

        kpt.eps_n = np.empty(2 * wfs.bd.mynbands)

        diagonalization_string = repr(self.diagonalizer)
        wfs.timer.start(diagonalization_string)
        from gpaw.utilities.lapack import general_diagonalize
        general_diagonalize(H2_MM, kpt.eps_n, S2_MM)
        kpt.C_nM = H2_MM
        #self.diagonalizer.diagonalize(H2_MM, kpt.C_nM, kpt.eps_n, S2_MM)
        wfs.timer.stop(diagonalization_string)
        kpt.C_nM.shape = (wfs.bd.mynbands * 4, M)
