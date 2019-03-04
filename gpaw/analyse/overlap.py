import numpy as np

from gpaw.utilities import packed_index

class Overlap:
    """Wave funcion overlap of two GPAW objects"""
    def __init__(self, calc):
        self.calc = calc
        self.nb, self.nk, self.ns = self.number_of_states(calc)
        self.gd = self.calc.wfs.gd
        self.kd = self.calc.wfs.kd

    def number_of_states(self, calc):
        return (calc.get_number_of_bands(), len(calc.get_ibz_k_points()),
                calc.get_number_of_spins())

    def pseudo(self, other, myspin=0, otherspin=0, normalize=True):
        """Overlap with pseudo wave functions only

        Parameter
        ---------
        other: gpaw
            gpaw-object containing pseudo wave functions
        normalize: bool
            normalize pseudo wave functions in the overlap integral

        Returns
        -------
        out: array
            u_kij =  \int dx mypsitilde_ki^*(x) otherpsitilde_kj(x)
        """
        nbo, nko, _ = self.number_of_states(other)
        assert(self.nk == nko)  # XXX allow for different number of k-points ?

        overlap_knn = []
        for k in range(self.nk):
            # XXX what if parallelized over spin or kpoints ?
            kpt_rank, u = self.kd.get_rank_and_index(myspin, k)
            assert self.kd.comm.rank == kpt_rank
            kpt_rank, uo = other.wfs.kd.get_rank_and_index(otherspin, k)
            assert self.kd.comm.rank == kpt_rank
            
            overlap_nn = np.zeros((self.nb, nbo), dtype=self.calc.wfs.dtype)
            mkpt = self.calc.wfs.kpt_u[u]
            okpt = other.wfs.kpt_u[uo]
            psit_nG = mkpt.psit_nG
            norm_n = self.gd.integrate(psit_nG.conj() * psit_nG)
            psito_nG = okpt.psit_nG
            normo_n = other.wfs.gd.integrate(psito_nG.conj() * psito_nG)
            for i in range(self.nb):
                p_nG = np.repeat(psit_nG[i].conj()[np.newaxis], nbo, axis=0)
                overlap_nn[i] = self.gd.integrate(p_nG * psito_nG)
                if normalize:
                    overlap_nn[i] /= np.sqrt(np.repeat(norm_n[i], nbo) *
                                             normo_n)
            overlap_knn.append(overlap_nn)
        return np.array(overlap_knn)

    def full(self, other, myspin=0, otherspin=0):
        """Overlap of Kohn-Sham states including local terms.

        Parameter
        ---------
        other: gpaw
            gpaw-object containing wave functions
 
        Returns
        -------
        out: array
            u_kij =  \int dx mypsi_ki^*(x) otherpsi_kj(x)
        """
        ov_knn = self.pseudo(other, normalize=False)
        for k in range(self.nk):
            # XXX what if parallelized over spin or kpoints ?
            kpt_rank, u = self.kd.get_rank_and_index(myspin, k)
            assert self.kd.comm.rank == kpt_rank
            kpt_rank, uo = other.wfs.kd.get_rank_and_index(otherspin, k)
            assert self.kd.comm.rank == kpt_rank

            mkpt = self.calc.wfs.kpt_u[u]
            okpt = other.wfs.kpt_u[uo]

            aov_nn = np.zeros_like(ov_knn[k])
            for a, mP_ni in mkpt.P_ani.items():
                oP_ni = okpt.P_ani[a]
                Delta_p = (np.sqrt(4 * np.pi) *
                           self.calc.wfs.setups[a].Delta_pL[:,0])
                for n0, mP_i in enumerate(mP_ni):
                    for n1, oP_i in enumerate(oP_ni):
                        ni = len(mP_i)
                        assert(len(oP_i) == ni)
                        for i, mP in enumerate(mP_i):
                            for j, oP in enumerate(oP_i):
                                ij = packed_index(i, j, ni)
                                aov_nn[n0, n1] += Delta_p[ij] * mP.conj() * oP
            self.calc.wfs.gd.comm.sum(aov_nn)
            ov_knn[k] += aov_nn
        return ov_knn
