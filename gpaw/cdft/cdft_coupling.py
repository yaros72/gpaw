'''A class for computing the
parameters for Marcus theory
from two constrained DFT
wave functions

Computes:
-the coupling matrix Hab
 Hab = <Psi_a|H|Psi_b>
-reorganization energy lambda
lambda = E_a(Rb)-E_a(Ra)
'''

import numpy as np
from gpaw.cdft.cdft import (WeightFunc, get_ks_energy_wo_external,
                    get_all_weight_functions)
from ase.units import kB as kb
from gpaw.utilities.ps2ae import PS2AE, interpolate_weight
from ase.units import Bohr
from ase.parallel import rank
import warnings
#from gpaw.utilities.blas import gemm
#from gpaw.utilities.tools import tri2full
#from gpaw.utilities import pack, unpack2

spin_state_error = ('The cDFT wave functions have\n'+
                    'different spin states! Similar\n'+
                    'spin states are required for coupling constant\n'+
                    'calculation!')

ae_ibz_error = ('The all electron calculation is unreliable with kpts\n'+
               'Please set AE to false')

nab_missing_error = ('The pair density matrix n_ab must\n'+
              'be provided for weight matrix calculation')
migliore_warning = ('WARNING! Migliore coupling might be unreliable!:\n'+
              '<F|GS> =<I|GS> and dE_if=0')

class CouplingParameters:

    def __init__(self, cdft_a=None, cdft_b=None,h=0.04,
               calc_a=None, calc_b=None, calc_gs=None,
               wfs_a='initial.gpw', wfs_b='final.gpw',
               AE=False, charge_difference=False,
               charge_regions_A=None, spin_regions_A=None,
               charge_regions_B = None, spin_regions_B=None,
               FA=None, FB=None,
               Va=None, Vb=None,
               NA=None, NB=None,
               n_ab=None, VW_ab=None, VW_ba=None,
               w_k=None, Rc={}, mu={},
               specific_bands_A=None, specific_bands_B=None,
               kpoint=None, spin=None,
               S_matrix = 'S_matrix', VW_matrix='VW_matrix',
               S=None, A=None, B=None,
               save_matrix=False,
               freq=None, temp=None, E_tst=None, reorg=None,
               use_adiabatic_correction=False, reaction_energy=0.,
               band_occupation_cutoff=0.5, energy_gap = None):

        '''cdft_a cdft_b: cdft calculators
        h = grid spacing in S_AB and W_AB calculations in AE mode
        AE = Use all electron wave functions


        if cdft calculators are not provided,
        calc_a, calc_b, wfs_a, wfs_b,gd,
        FA,FB,Va, Vb, NA, NB,
        charge_regions_A/B must be provided

        calc_a/b = GPAW calculators for a/b states
        wfs_a/b = wfs of the a/b states
        FA/FB = pseudo cdft free energies of A/B states in eV
                as from cdft.free_energy() or cdft.Ecdft

        Va/b = list of lagrange multipliers in eV
        NA/NB = charge + spin constraints in A/B states
             !NOTE! Total number of electrons as from cdft.get_constraints()

        weight_A/B = the weight functions for A/B
            Array

        N_charge_regions_A/B = number of constrained charges
            int

        E_KS_A/B = KS energies with out the external potential
                  (Ekin+Epot+Ebar+Exc+S) in eV

        n_ab = S_AB_ij(k) = <A_i(k)|B_j(k)>

        VW_ab,VW_ba = <A(k)|sum_c Vc_B*Wc_B|B(k)>, <B(k)|sum_c Vc_A*Wc_A|B(k)>

        WAB, WBA = sum_k sum_ij sum_c <A_i(k)|Wc_B|B_j(k)>

        w_k = combined kpt weight, (w_kA*w_kB) in eq 48 and 49
              can be provided to compute overlap and
              weight matrices

        specific_bands, kpoint, spin = [[],[]], int, int
            allows to extract density matrix elements
            and weight-density matrix elements for
            specified orbitals Eq. 48 and 49
            [[],[]] = indices of first for alpha,
                   second for beta spin bands

            If one these is activated, matrix of
            int[psi_sk^i*psi_sk^j] is returned instead
            of S_ab or W_ab

        S_matrix, W_matrix = str
            place to print density and
            weight-density matrix
            Turned on automatically if bands,
            spin, or kpoint are specified

        S, A, B = np.array, stored migliore matrices

        for calculation of electron transfer rates
        using the Marcus theory the following can be
        specified:

        freq = effective frequency
        temp = temperature
        E_tst = adiabatic reaction barrier
        use_adiabatic_correction = False/True
            add the adiabadicity correction to
            Marcus barrier (can some times give very
            too small barriers for an adiabatic reaction
            with large coupling...)
        reaction_energy: E_B(RB)-E_A(RA)
        band_occupation_cutoff = minimum filling of band
            to be included in the wfs
        Rc, mu: Gaussian parameters. If not provided either standard or
            the ones from a cdft will be used
        '''

        self.charge_difference = charge_difference
        self.AE = AE

        if cdft_a is not None and cdft_b is not None:
            self.cdft_A = cdft_a
            self.cdft_B = cdft_b
            self.Rc = self.cdft_A.Rc
            self.mu = self.cdft_A.mu

            self.calc_A = self.cdft_A.calc
            self.calc_B = self.cdft_B.calc

            if self.AE:
                self.gd = self.cdft_A.get_grid()
            else:
                self.gd = self.cdft_A.calc.density.gd

            # cDFT free energies
            self.FA = self.cdft_A.cdft_free_energy()
            self.FB = self.cdft_B.cdft_free_energy()

            # cDFT energies
            self.EA = self.cdft_A.dft_energy()
            self.EB = self.cdft_B.dft_energy()

            # lagrange multipliers
            self.Va = self.cdft_A.get_lagrangians()
            self.Vb = self.cdft_B.get_lagrangians()

            # constraint values
            self.NA = self.cdft_A.get_constraints()
            self.NB = self.cdft_B.get_constraints()

            # number of charge regions
            self.n_charge_regionsA = self.cdft_A.n_charge_regions
            self.n_charge_regionsB = self.cdft_B.n_charge_regions
            self.regionsA = self.cdft_A.regions
            self.regionsB = self.cdft_B.regions
            self.atoms = self.calc_A.get_atoms()

            if self.AE:
                self.fineweightA = self.cdft_A.get_weight(save=False, pad=True)
                self.fineweightB = self.cdft_B.get_weight(save=False, pad=True)
            else:
                self.coarseweightA = get_all_weight_functions(self.calc_A.atoms,
                            self.gd, self.regionsA, self.charge_difference,
                            self.Rc, self.mu)
                self.coarseweightB = get_all_weight_functions(self.calc_B.atoms,
                            self.gd, self.regionsB, self.charge_difference,
                            self.Rc, self.mu)

        else:
            self.calc_A = calc_a
            self.calc_B = calc_b

            if self.AE:
                self.gd = self.calc_A.density.finegd
            else:
                self.gd = self.calc_A.density.gd

            # cDFT free energies
            self.FA = FA
            self.FB = FB

            # Weights, i.e., lagrange multipliers
            self.Va = Va
            self.Vb = Va

            regionsA = []
            if charge_regions_A is not None:
                regionsA.append(charge_regions_A)
            if spin_regions_A is not None:
                regionsA.append(spin_regions_A)

            regionsB = []
            if charge_regions_B is not None:
                regionsB.append(charge_regions_B)
            if spin_regions_B is not None:
                regionsB.append(spin_regions_B)

            self.regionsA = regionsA
            self.regionsB = regionsB
            self.NA = NA
            self.NB = NB

            # see if a correct set of Rc and mu are provided
            if (set(Rc) == set(self.calc_A.atoms.get_chemical_symbols()) and
                set(mu) == set(self.calc_A.atoms.get_chemical_symbols())):
                self.Rc = Rc
                self.mu = mu
            else:
                # get mu and Rc the long way
                w_temp = WeightFunc(gd=self.gd, atoms=self.calc_a.atoms,
                    indices=self.regionsA, Rc=Rc, mu=mu)
                self.Rc, self.mu = w_temp.get_Rc_and_mu()
                del w_temp

            if self.AE:
                self.fineweightA = get_all_weight_functions(self.calc_a.atoms,
                            self.gd, self.regionsA, self.charge_difference,
                            self.Rc, self.mu)

                self.fineweightB = get_all_weight_functions(self.calc_b.atoms,
                            self.gd, self.regionsB, self.charge_difference,
                            self.Rc, self.mu)

            else:
                self.coarseweightA = get_all_weight_functions(self.calc_a.atoms,
                            self.gd, self.regionsA, self.charge_difference,
                            self.Rc, self.mu)
                self.coarseweightB = get_all_weight_functions(self.calc_b.atoms,
                            self.gd, self.regionsB, self.charge_difference,
                            self.Rc, self.mu)


            self.n_charge_regionsA = len(charge_regions_A)
            self.n_charge_regionsB = len(charge_regions_B)

            # Do not include external potential
            self.EA = get_ks_energy_wo_external(self.calc_A)
            self.EB = get_ks_energy_wo_external(self.calc_B)

        self.finegd = self.calc_A.density.finegd
        self.energy_gap = energy_gap
        self.Va = np.asarray(self.Va)
        self.Vb = np.asarray(self.Vb)

        # wave functions
        if wfs_a is not None and wfs_b is not None:
            self.wfs_A = wfs_a
            self.wfs_B = wfs_b
        else:
            self.wfs_A = self.calc_A.wfs
            self.wfs_B = self.calc_B.wfs

        # ground state calculator for migliore

        if calc_gs is not None:
            self.calc_gs = calc_gs
        if S is not None:
            self.S = np.load(S)
        if A is not None:
            self.A = np.load(A)
        if B is not None:
            self.B = np.load(B)
        if w_k is not None:
            self.w_AB = self.w_AC = self.w_BC = np.load(w_k)

        # use precalculated pair density/weight matrices
        if w_k:
            self.w_k = np.load(w_k)
        if n_ab:
            self.n_ab = np.load(n_ab)

        if VW_ab:
            self.VW_AB = np.load(VW_ab)
            self.VW_BA = np.load(VW_ba)

        # specify bands to be treated
        self.specific_bands_A = specific_bands_A
        self.specific_bands_B = specific_bands_B

        if kpoint:
            self.kpoints = kpoint
        else:
            self.kpoints = self.calc_A.wfs.kd.nibzkpts
        self.spin = spin

        # where to save matrices
        self.S_matrix = S_matrix
        self.VW_matrix = VW_matrix

        self.save_matrix = save_matrix
        # Only part of bands treated --> matrices saved
        if self.specific_bands_A or self.specific_bands_B or self.spin:
            self.save_matrix = True

        self.H_AA = self.EA
        self.H_BB = self.EB

        self.density = self.calc_A.density.finegd.zeros()
        self.atoms = self.calc_A.atoms
        self.natoms = len(self.atoms)
        self.h = h # all-electron interpolation grid spacing

        # controls for rate calculation

        self.E_tst = E_tst # adiabatic barrier

        # effective frequency
        if freq:
            self.freq = freq
        else:
            self.freq = 1.e12
        # temperature
        if temp:
            self.temp = temp
        else:
            self.temp = 298
        self.reorg = reorg

        self.reaction_energy = reaction_energy
        self.use_adiabatic_correction = use_adiabatic_correction
        self.band_occupation_cutoff = band_occupation_cutoff

    def get_coupling_term(self):
        # coupling by Migliore
        # dx.doi.org/10.1021/ct200192d, eq 6
        # check that needed terms exist

        if (hasattr(self,'H') and
            hasattr(self,'S')):
            if rank == 0:
                H = self.H.copy()
                S = self.S.copy()

        else:
            self.make_hamiltonian_matrix()
            if rank == 0:
                H = self.H.copy()
                S = self.S.copy()
        if rank == 0:
            H_av = 0.5*(H[0][1] + H[1][0])
            S_av = 0.5*(S[0][1] + S[1][0])

            self.ct = (1./(1 - S_av**2)) * np.abs(H_av -
                       S_av * (H[0][0] + H[1][1])/2. )
            return self.ct

    def get_coupling_term_from_lowdin(self):
        self.make_hamiltonian_matrix()
        if rank == 0:
            H_orth = self.lowdin_orthogonalize_cdft_hamiltonian()
            self.ct = 1./2.*np.real(H_orth[0][1] + H_orth[1][0])
            return self.ct

    def lowdin_orthogonalize_cdft_hamiltonian(self):
        # H_orth = S^(-1/2)*H_cdft*S^(-1/2)

        U,D,V =np.linalg.svd(self.S, full_matrices=True) #V=U(+)
        s = np.diag(D**(-0.5))
        S_square = np.dot(U,np.dot(s,V)) # S^(-1/2)

        H_orth = np.dot(S_square,np.dot(self.H, S_square))
        return H_orth

    def get_migliore_coupling(self):
        # the entire migliore method
        # from dx.doi.org/10.1021/ct200192d,
        # requires also ground state adiabatic wf
        if not hasattr(self,'S') or not hasattr(self,'A') or not hasattr(self, 'B'):
            self.S, self.A, self.B, self.w_AB, self.w_AC, self.w_BC = self.get_migliore_wf_overlaps(self.calc_A,
                self.calc_B, self.calc_gs)
        if rank == 0:
            SIF = 0.
            SFI = 0.
            A = 0.
            B = 0.

            for k in range(len(self.w_AB)):
                SIF += self.w_AB[k] * np.linalg.det(self.S[k])
                SFI += self.w_AB[k] * np.linalg.det(np.transpose(np.conjugate(self.S[k])))
            for k in range(len(self.w_AC)):
                A += self.w_AC[k] * np.linalg.det(self.A[k])
            for k in range(len(self.w_BC)):
                B += self.w_BC[k] * np.linalg.det(self.B[k])

            if self.energy_gap is None:
                self.get_energy_gap()
            dE_if = self.energy_gap
            if np.abs(dE_if) < 0.001 and np.abs(A**2-B**2) < 0.001:
                warnings.warn(migliore_warning)
            self.ct = np.abs( (A*B/(A**2-B**2)) * dE_if * \
                      (1.-SIF*(A**2+B**2)/(2*A*B)) * 1./(1-SIF**2) )
            return self.ct

    def make_hamiltonian_matrix(self):
        # this returns a 2x2 cDFT Hamiltonian
        # in a non-orthogonal basis -->
        # made orthogonal in get_coupling_term
        #self.get_diagonal_H_elements()
        H_AA = self.H_AA # diabat A with H^KS_A
        H_BB = self.H_BB # diabat B with H^KS_B

        if hasattr(self, 'S'):
            S = self.S
        else:
            S = self.get_overlap_matrix()
            self.S = S

        if hasattr(self,'VW'):
            VW = self.VW
        else:
            VW = self.get_weight_matrix()
        if rank == 0:
            S_AB = S[0][1]
            S_BA = S[1][0]

            h_AB = self.FB*S_AB - VW[0][1]
            h_BA = self.FA*S_BA - VW[1][0]

            # Ensure that H is hermitian
            H_AB = 1./2. * (h_AB + h_BA)
            H_BA = 1./2. * (h_AB + h_BA).conj()

            self.H = np.array([[H_AA, H_BA],[H_AB,H_BB]])
            return self.H

    def get_ae_pair_weight_matrix(self):
        if self.calc_A.wfs.kd.nibzkpts != 1:
            raise ValueError(ae_ibz_error)
        if hasattr(self, 'n_ab'):
            pass
        else:
           raise ValueError(nab_missing_error)

        # pseudo wfs to all-electron wfs
        psi_A = PS2AE(self.calc_A, h=self.h)
        psi_B = PS2AE(self.calc_B, h=self.h)

        ns = self.calc_A.wfs.nspins

        # weight functions sum_c VcWc
        wa = []
        wb = []

        for a in range(len(self.Va)):
            wa.append(interpolate_weight(self.calc_A,
                    self.fineweightA[a], h=self.h))
        for b in range(len(self.Vb)):
            wb.append(interpolate_weight(self.calc_B,
                    self.fineweightB[b], h=self.h))
        wa = np.asarray(wa)
        wb = np.asarray(wb)

        # check number of occupied and total number of bands
        n_occup_A = self.get_n_occupied_bands(self.calc_A)
        n_occup_B = self.get_n_occupied_bands(self.calc_B)

        # place to store <i_A(k)|w|j_B(k)>
        w_kij_AB = []
        w_kij_BA = []

        # k-point weights
        w_k = np.zeros(self.calc_A.wfs.kd.nibzkpts,dtype=np.float32)

        # get weight matrix at for each ij band at kpt and spin
        # the resulting matrix is organized in alpha and beta blocks
        #        |       |    |
        #        |  a    | 0  |    a:<psi_a|Vb*wb|psi_a> != 0
        # VW_AB =|_______|____|  , <psi_a|w|psi_b> = 0
        #        |   0   |  b |    b:<psi_b|Vb*wb|psi_b> != 0
        #        |       |    |
        #
        # a = nAa x nAa, b = nAb x nAb

        for spin in range(ns):
            for k in range(self.calc_A.wfs.kd.nibzkpts):

                # k-dependent overlap/pair density matrices
                inv_S = np.linalg.inv(self.n_ab[k])
                det_S = np.linalg.det(self.n_ab[k])
                I = np.identity(inv_S.shape[0])
                C_ab = np.transpose(np.dot(inv_S, (det_S*I)))

                nAa, nAb, nBa, nBb = self.check_bands(n_occup_A, n_occup_B, k)
                nas, n_occup, n_occup_s = self.check_spin_and_occupations(nAa,
                                                                nAb, nBa, nBb)

                # check that a and b cDFT states have similar spin state
                if np.sign(nAa-nAb) != np.sign(nBa-nBb):
                    warning= UserWarning(spin_state_error)
                    warnings.warn(warning)
                # form overlap matrices of correct size for each kpt

                if spin == 0:
                    w_kij_AB.append(np.zeros((n_occup,n_occup),dtype = np.complex))
                    w_kij_BA.append(np.zeros((n_occup,n_occup), dtype = np.complex))

                # store k-point weights
                kd = self.calc_A.wfs.kd
                w_kA = kd.weight_k[k]
                kd = self.calc_B.wfs.kd
                w_kB = kd.weight_k[k]

                for i in range(n_occup_s[spin]):
                    for j in range(n_occup_s[spin]):
                        I = spin*nas + i
                        J = spin*nas + j

                        psi_kA = psi_A.get_wave_function(n=i, k=k, s=spin, ae=True)
                        psi_kB = psi_B.get_wave_function(n=j, k=k, s=spin, ae=True)
                        w_ij_AB = []
                        w_ji_BA = []

                        for b in range(len(self.Vb)):
                            integral = psi_B.gd.integrate(psi_kA.conj() * wb[b] * psi_kB,
                                global_integral=True) * C_ab[I][J]

                            if b >= self.n_charge_regionsB and spin == 1:
                                # for charge constraint w > 0
                                integral *= -1.
                            w_ij_AB.append(-integral)

                        for a in range(len(self.Va)):
                            integral = psi_A.gd.integrate( psi_kB.conj() * wa[a] * psi_kA,
                                        global_integral=True) * C_ab[J][I]
                            if a >= self.n_charge_regionsA and spin == 1:
                                integral *= -1.
                            w_ji_BA.append(-integral)

                        w_ij_AB = np.asarray(w_ij_AB)*Bohr**3
                        w_ji_BA = np.asarray(w_ji_BA)*Bohr**3

                        # collect kpt weight, only once per kpt
                        if spin == 0 and i == 0 and j == 0:
                            w_k[k] = (w_kA + w_kB)/2.

                        w_kij_AB[k][I][J] += np.dot(self.Vb, w_ij_AB).sum()
                        w_kij_BA[k][J][I] += np.dot(self.Va, w_ji_BA).sum()

        self.w_k = w_k
        self.VW_AB = w_kij_AB
        self.VW_BA = w_kij_BA

        if self.save_matrix:
            np.save(self.VW_matrix + 'final_AB',self.VW_AB)
            np.save(self.VW_matrix + 'final_BA',self.VW_BA)
        return self.VW_AB, self.VW_BA, self.w_k

    def get_pair_weight_matrix(self):
        # <Psi_A|Psi_B> using pseudo wave functions and atomic corrections

        if not hasattr(self, 'n_ab'):
            raise ValueError(nab_missing_error)

        # check number of occupied and total number of bands
        n_occup_A = self.get_n_occupied_bands(self.calc_A) # total of filled a and b bands
        n_occup_B = self.get_n_occupied_bands(self.calc_B)

        # place to store <i_A(k)|w|j_B(k)>
        w_kij_AB = []
        w_kij_BA = []
        # k-point weights
        w_k = np.zeros(self.calc_A.wfs.kd.nibzkpts)

        # get weight matrix at for each ij band at kpt and spin
        # the resulting matrix is organized in alpha and beta blocks
        #        |       |    |
        #        |  a    | 0  |    a:<psi_a|Vb*wb|psi_a> != 0
        # VW_AB =|_______|____|  , <psi_a|w|psi_b> = 0
        #        |   0   |  b |    b:<psi_b|Vb*wb|psi_b> != 0
        #        |       |    |
        #
        # a = nAa x nAa, b = nAb x nAb

        for kpt_a, kpt_b in zip(self.calc_A.wfs.kpt_u, self.calc_B.wfs.kpt_u):
            k = kpt_a.k
            spin = kpt_a.s

            # k-dependent overlap/pair density matrices
            inv_S = np.linalg.inv(self.n_ab[k])
            det_S = np.linalg.det(self.n_ab[k])
            I = np.identity(inv_S.shape[0])
            C_ab = np.transpose(np.dot(inv_S, (det_S*I)))

            inv_S = np.linalg.inv(np.transpose(self.n_ab[k]).conj())
            det_S = np.linalg.det(np.transpose(self.n_ab[k]))
            I = np.identity(inv_S.shape[0])
            C_ba = np.transpose(np.dot(inv_S, (det_S*I)))

            nAa, nAb, nBa, nBb = self.check_bands(n_occup_A, n_occup_B, k)
            nas, n_occup, n_occup_s = self.check_spin_and_occupations(nAa,
                                                            nAb, nBa, nBb)

            # check that a and b cDFT states have similar spin state
            if np.sign(nAa-nAb) != np.sign(nBa-nBb):
                warning= UserWarning(spin_state_error)
                warnings.warn(warning)
            # form overlap matrices of correct size for each kpt

            if spin == 0:
                w_kij_AB.append(np.zeros((n_occup,n_occup),dtype = np.complex))
                w_kij_BA.append(np.zeros((n_occup,n_occup), dtype = np.complex))

            # store k-point weights
            kd = self.calc_A.wfs.kd
            w_kA = kd.weight_k[k]
            kd = self.calc_B.wfs.kd
            w_kB = kd.weight_k[k]

            for b in range(len(self.Vb)):
                self.get_matrix_element(kpt_a.psit_nG, kpt_a.P_ani,
                                   kpt_b.psit_nG, kpt_b.P_ani,
                                   n_occup_s ,n_occup_s,
                                   w_kij_AB, self.regionsB[b],
                                   k, spin, nas, V=self.Vb[b],
                                   W=self.coarseweightB[b], C12=C_ab)

                if b >= self.n_charge_regionsB and spin == 1:
                    # change sign of b spin constraint
                    w_kij_AB[k][nas:,nas:] *= -1.

            for a in range(len(self.Va)):
                self.get_matrix_element(kpt_b.psit_nG, kpt_b.P_ani,
                                   kpt_a.psit_nG, kpt_a.P_ani,
                                   n_occup_s ,n_occup_s,
                                   w_kij_BA, self.regionsA[a],
                                   k, spin, nas, V=self.Va[a],
                                   W=self.coarseweightA[a], C12=C_ba)
                if a >= self.n_charge_regionsA and spin == 1:
                    w_kij_BA[k][nas:,nas:] *= -1.


            if spin == 0:
                w_k[k] = (w_kA + w_kB)/2.

        self.VW_BA = np.asarray(w_kij_BA)
        self.VW_AB = np.asarray(w_kij_AB)
        self.w_k = w_k

        return self.VW_AB, self.VW_BA, self.w_k

    def get_weight_matrix(self):
        # from the pair density matrix
        if not (hasattr(self, 'VW_AB')):
            if self.AE:
                self.get_ae_pair_weight_matrix()
            else:
                self.get_pair_weight_matrix()
        if not (hasattr(self,'w_k')):
            self.get_ae_pair_density_matrix()

        if rank == 0:
            # diagonal of V*weight matrix
            self.VW = np.zeros((2,2))
            # fill diagonal

            self.VW[0][0] += np.sum(self.NA)
            self.VW[1][1] += np.sum(self.NB)

            W_k_AB = np.zeros(len(self.w_k))
            W_k_BA = np.zeros(len(self.w_k))

            for k in range(len(self.w_k)):
                # sum_k (sum_ij <i|sum_c Vc*wc| j> * C_ij
                W_k_AB[k] = self.VW_AB[k].sum()
                W_k_AB[k] *= self.w_k[k]
                W_k_BA[k] = self.VW_BA[k].sum()
                W_k_BA[k] *= self.w_k[k]

            self.VW[0][1] = W_k_AB.sum()
            self.VW[1][0] = W_k_BA.sum()

            return self.VW

    def get_ae_pair_density_matrix(self,calc_A, calc_B, matrix_name=None):
        if calc_A.wfs.kd.nibzkpts != 1:
           raise ValueError(ae_ibz_error)
        # <Psi_A|Psi_B> using the all-electron pair density
        psi_A = PS2AE(calc_A, h=self.h)
        psi_B = PS2AE(calc_B, h=self.h)

        ns = calc_A.wfs.nspins

        # total of filled a and b bands for each spin and kpt
        n_occup_A = self.get_n_occupied_bands(calc_A)
        n_occup_B = self.get_n_occupied_bands(calc_B)

        # list to store k-dependent pair density
        n_AB = []
        w_k = np.zeros(calc_A.wfs.kd.nibzkpts) #store kpt weights

        # get overlap at for each ij band at kpt and spin
        # the resulting matrix is organized in alpha and beta blocks
        #    |       |    |
        #    |  a    | 0  |    a:<psi_a|psi_a> != 0
        # S =|_______|____|  , <psi_a|psi_b> = 0
        #    |   0   |  b |    b:<psi_b|psi_b> != 0
        #    |       |    |
        #
        # a = naa x naa, b = nab x nab

        for spin in range(ns):
            for k in range(calc_A.wfs.kd.nibzkpts):

                nAa, nAb, nBa, nBb = self.check_bands(n_occup_A, n_occup_B, k)
                nas, n_occup, n_occup_s = self.check_spin_and_occupations(nAa,
                                                                nAb, nBa, nBb)
                kd = calc_A.wfs.kd
                w_kA = kd.weight_k[k]
                kd = calc_B.wfs.kd
                w_kB = kd.weight_k[k]

                # check that a and b cDFT states have similar spin state
                if np.sign(nAa-nAb) != np.sign(nBa-nBb):
                    warning= UserWarning('The cDFT wave functions have\n'
                        'different spin states! Similar\n'
                        'spin states are required for coupling constant\n'
                        'calculation!')
                    warnings.warn(warning)
                # form overlap matrices of correct size for each kpt
                if spin == 0:
                    n_AB.append(np.zeros((n_occup,n_occup),dtype = np.complex))

                for i in range(n_occup_s[spin]):
                    for j in range(n_occup_s[spin]):
                        # take only the bands which contain electrons in spin-orbital
                        psi_kA = psi_A.get_wave_function(n=i, k=k, s=spin, ae=True)
                        psi_kB = psi_B.get_wave_function(n=j, k=k, s=spin, ae=True)
                        n_ij = psi_A.gd.integrate(psi_kA.conj(), psi_kB, global_integral=True)

                        if spin == 0 and i == 0 and j == 0:
                            w_k[k] = (w_kA + w_kB)/2.

                        I = spin*nas + i
                        J = spin*nas + j
                        n_AB[k][I][J] = n_ij * Bohr**3

        n_AB = np.asarray(n_AB)
        self.w_k = w_k
        self.n_ab = n_AB

        if self.save_matrix:
            if matrix_name is None:
                np.save(self.S_matrix+'final', self.n_ab)
            else:
                np.save('%s_final' % matrix_name, self.n_ab)
        return self.n_ab, self.w_k

    def get_pair_density_matrix(self, calc_A, calc_B, matrix_name=None):
        # <Psi_A|Psi_B> using pseudo wave functions and atomic corrections
        assert calc_A.wfs.kd.nibzkpts == calc_B.wfs.kd.nibzkpts

        # total of filled a and b bands for each spin and kpt
        n_occup_A = self.get_n_occupied_bands(calc_A)
        n_occup_B = self.get_n_occupied_bands(calc_B)

        # list to store k-dependent pair density
        n_AB = []

        w_k = np.zeros(calc_A.wfs.kd.nibzkpts) #store kpt weights

        # get overlap at for each ij band at kpt and spin
        # the resulting matrix is organized in alpha and beta blocks
        #    |       |    |
        #    |  a    | 0  |    a:<psi_a|psi_a> != 0
        # S =|_______|____|  , <psi_a|psi_b> = 0
        #    |   0   |  b |    b:<psi_b|psi_b> != 0
        #    |       |    |
        #
        # a = naa x naa, b = nab x nab

        for kpt_a, kpt_b in zip(calc_A.wfs.kpt_u, calc_B.wfs.kpt_u):

            k = kpt_a.k
            spin = kpt_a.s
            nAa, nAb, nBa, nBb = self.check_bands(n_occup_A, n_occup_B, k)
            nas, n_occup, n_occup_s = self.check_spin_and_occupations(nAa, nAb, nBa, nBb)

            # check that a and b cDFT states have similar spin state
            if np.sign(nAa-nAb) != np.sign(nBa-nBb):
                warning= UserWarning('The cDFT wave functions have\n'
                    'different spin states! Similar\n'
                    'spin states are required for coupling constant\n'
                    'calculation!')
                warnings.warn(warning)
            # form overlap matrices of correct size for each kpt
            if spin == 0:
                n_AB.append(np.zeros((n_occup,n_occup), dtype=np.complex))

            kd = calc_A.wfs.kd
            w_kA = kd.weight_k[k]
            kd = calc_B.wfs.kd
            w_kB = kd.weight_k[k]

            self.get_matrix_element(kpt_b.psit_nG, kpt_b.P_ani,
                            kpt_a.psit_nG, kpt_a.P_ani,
                            n_occup_s, n_occup_s,
                            n_AB, None,
                            k, spin, nas)

            if spin == 0:
                w_k[k] = (w_kA + w_kB)/2.
        n_AB = np.asarray(n_AB)
        self.w_k = w_k
        self.n_ab = n_AB

        if self.save_matrix:
            if matrix_name is None:
                np.save(self.S_matrix+'final', self.n_ab)
            else:
                np.save('%s_final' % matrix_name, self.n_ab)

        return self.n_ab, self.w_k

    def get_overlap_matrix(self, save=False, name='wf_overlap'):

        # from the pair density matrix
        if not (hasattr(self,'n_ab')):
            if self.AE:
                self.n_ab, self.w_k = self.get_ae_pair_density_matrix(self.calc_A, self.calc_B)
            else:
                self.n_ab, self.w_k = self.get_pair_density_matrix(self.calc_A, self.calc_B)

        if not (hasattr(self,'w_k')):
            self.get_ae_pair_density_matrix(self.calc_A, self.calc_B)
        if rank == 0:
            self.S = np.identity(2)

            S_k_AB = np.zeros(len(self.w_k))
            S_k_BA = np.zeros(len(self.w_k))

            for k in range(len(self.w_k)):
                S_k_AB[k] = self.w_k[k] * np.linalg.det(self.n_ab[k])
                # determinant of the complex conjugate
                S_k_BA[k] = self.w_k[k] * np.linalg.det(np.transpose(self.n_ab[k]).conj())

            S_AB = S_k_AB.sum()
            S_BA = S_k_BA.sum()

            # fill 2x2 overlap matrix
            self.S[0][1] = S_AB
            self.S[1][0] = S_BA

            if save:
                np.save('%s' % name,self.S)
            return self.S

    def get_migliore_wf_overlaps(self, calc_A, calc_B, calc_gs):
        '''A = <I|GS>, B= <F|GS>, S = <I|F> '''
        if self.AE:
            S, w_AB = self.get_ae_pair_density_matrix(calc_A, calc_B, matrix_name='Sif')
            A, w_AC = self.get_ae_pair_density_matrix(calc_A, calc_gs, matrix_name='A')
            B, w_BC = self.get_ae_pair_density_matrix(calc_B, calc_gs, matrix_name='B')
        else:
            S, w_AB = self.get_pair_density_matrix(calc_A, calc_B, matrix_name='Sif')
            A, w_AC = self.get_pair_density_matrix(calc_A, calc_gs, matrix_name='A')
            B, w_BC = self.get_pair_density_matrix(calc_B, calc_gs, matrix_name='B')

        return S, A, B, w_AB, w_AC, w_BC

    def get_matrix_element(self, psit1_nG, P1_ani, psit2_nG, P2_ani,
            n_occup_1 ,n_occup_2, vw, region, k, s, nas, V=1., W=1., C12=None):
        # weight: V*W acting on psitb_nG: VW_ij = <psita_nG_i|VW|psitb_nG_j> or
        # overlap <psita_nG_i|psitb_nG_j>
        # includes occupation dependency and only states with filled
        # orbitals are considered
        # Cab = cofactor matrix
        # k and s --> kpt and spin

        VW_ij = np.zeros((len(psit1_nG), len(psit2_nG)))
        VW = V*W

        for n1 in range(len(psit1_nG)):
            for n2 in range(len(psit2_nG)):
                nijt_G = np.multiply(psit1_nG[n1].conj(), psit2_nG[n2])
                VW_ij[n1][n2] = self.gd.integrate(VW * nijt_G,
                                global_integral=True)

        P_array = np.zeros(VW_ij.shape)
        for a, P1_ni in P1_ani.items():
            P2_ni = P2_ani[a]
            if region is None or a in region:
                # the atomic correction is zero outside
                # the augmentation regions --> w=0 if
                # a is not in the w.
                # region = None --> overlap and all terms are used
                inner = np.dot(P1_ni, self.calc_A.wfs.setups[a].dO_ii)
                outer = (np.dot(P2_ni,V * np.conjugate(inner.T))).T
                P_array += outer

        self.calc_A.density.gd.comm.sum(P_array)
        VW_ij += P_array

        for i,j in enumerate(range(n_occup_1[s])):
            for x,y in enumerate(range(n_occup_2[s])):
                I = s*nas + i
                X = s*nas + x
                if C12 is None:
                    vw[k][I][X] += VW_ij[j][y]
                else:
                    vw[k][I][X] += VW_ij[j][y]*C12[I][X]

    def get_reorganization_energy(self):
        # get Ea (Rb) - Ea(Ra) -->
        # cdft energy at geometry Rb
        # with charge constraint A
        geometry = self.calc_B.get_atoms()

        cdft = self.cdft_A
        # set cdft_a on geometry of B
        geometry.set_calculator(cdft)
        self.reorg = geometry.get_potential_energy()
        self.reorg -= self.EA
        return self.reorg

    def get_energy_gap(self):
        # get Eb (Ra) - Ea(Ra) -->
        # cdft energy at geometry Ra
        # with charge constraint A or B
        geometry = self.calc_A.get_atoms()

        cdft = self.cdft_B
        # set cdft_a on geometry of B
        geometry.set_calculator(cdft)
        self.energy_gap = geometry.get_potential_energy()
        self.energy_gap -= self.EA
        return self.energy_gap

    def get_landau_zener(self):
        # computes the Landau-Zener factor

        planck = 4.135667e-15 #eV s

        if self.reorg is not None:
            Lambda = self.reorg
        else:
            Lambda = self.get_reorganization_energy()

        if hasattr(self, 'ct'):
            Hab = self.ct
        else:
            Hab = self.get_coupling_term()

        self.P_lz = 1 - np.exp(-np.pi**(3./2.) * (np.abs(Hab))**2 / \
                   (planck * self.freq * np.sqrt(self.temp*Lambda*kb)))

        return self.P_lz

    def get_marcus_rate(self):
        # adiabatic transition state energy
        if self.E_tst:
            E_tst = self.E_tst
        else:
            # compute the activation energy
            # from the classical marcus
            # parabolas
            E_tst = self.get_marcus_barrier()

        # electron transmission coeffiecient
        # is the reaction diabatic or adiabatic?
        P_lz = self.get_landau_zener()
        dE = self.EA - self.EB
        if dE >= -self.reorg:
            # normal
            kappa = 2. * P_lz / (1 + P_lz)
        else:
            kappa = 2 * P_lz /(1 - P_lz)
        rate = kappa * self.freq * np.exp(-E_tst/(kb * self.temp))

        return rate

    def get_marcus_barrier(self):
        # approximate barrier from
        # two marcus parabolas
        # and an adiabatic correction

        # reaction energy
        dE = self.reaction_energy

        # crossing of the parabolas
        barrier = 1. / (4. * self.reorg) * \
                   (self.reorg + dE)**2
        if self.use_adiabatic_correction:
            # adiabatic correction
            correction = np.abs(self.ct) + \
                (self.reorg + dE) / 2. -\
                np.sqrt((1. / 4. * (self.reorg +self.reorg))**2 + \
                (np.abs(self.ct))**2)

            return barrier - correction
        else:
            return barrier

    def get_n_occupied_bands(self, calc):
        ''' how many occupied bands?'''

        ns = calc.wfs.nspins
        occup_ks = np.zeros((len(calc.wfs.kd.weight_k),ns), dtype=int)
        w = calc.wfs.kd.weight_k
        for k in range(calc.wfs.kd.nibzkpts):
            for s in range(2):
                f_n = calc.get_occupation_numbers(kpt=k, spin=s)
                f_n *= 1./w[k]
                f_N = f_n > self.band_occupation_cutoff
                occup_ks[k][s] += f_N.sum()
        return occup_ks

    def check_bands(self, n_occup_A, n_occup_B, k):
        if not self.specific_bands_A and not self.specific_bands_B:
            nAa, nAb = n_occup_A[k][0], n_occup_A[k][1]
            nBa, nBb = n_occup_B[k][0], n_occup_B[k][1]
        else:# choose subset of orbitals for coupling
            if self.specific_bands_A:
                nAa, nAb = self.specific_bands_A[0], self.specific_bands_A[1]
            if self.specific_bands_B:
                nBa, nBb= self.specific_bands_B[0], self.specific_bands_B[1]

        return nAa, nAb, nBa, nBb

    def check_spin_and_occupations(self, nAa, nAb, nBa, nBb):
        nas = np.max((nAa,nBa))
        nbs = np.max((nAb,nBb))
        n_occup_s = [nas,nbs]
        n_occup = sum(n_occup_s)

        return nas, n_occup, n_occup_s
