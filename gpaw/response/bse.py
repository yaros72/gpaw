from __future__ import print_function

import functools
from time import time, ctime
from datetime import timedelta

import sys
from os import path as op

import numpy as np
from ase.units import Hartree, Bohr
from ase.utils import devnull
from ase.dft import monkhorst_pack

from gpaw import GPAW
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.spinorbit import get_spinorbit_eigenvalues

from gpaw.blacs import BlacsGrid, Redistributor
from gpaw.mpi import world, serial_comm
from gpaw.response.chi0 import Chi0
from gpaw.response.kernels import get_coulomb_kernel
from gpaw.response.kernels import get_integrated_kernel
from gpaw.response.wstc import WignerSeitzTruncatedCoulomb
from gpaw.response.pair import PairDensity


class BSE:
    def __init__(self,
                 calc=None,
                 spinors=False,
                 ecut=10.,
                 scale=1.0,
                 nbands=None,
                 valence_bands=None,
                 conduction_bands=None,
                 eshift=None,
                 gw_skn=None,
                 truncation=None,
                 integrate_gamma=1,
                 txt=sys.stdout,
                 mode='BSE',
                 wfile=None,
                 write_h=False,
                 write_v=False):

        """Creates the BSE object

        calc: str or calculator object
            The string should refer to the .gpw file contaning KS orbitals
        ecut: float
            Plane wave cutoff energy (eV)
        nbands: int
            Number of bands used for the screened interaction
        valence_bands: list
            Valence bands used in the BSE Hamiltonian
        conduction_bands: list
            Conduction bands used in the BSE Hamiltonian
        eshift: float
            Scissors operator opening the gap (eV)
        gw_skn: list / array
            List or array defining the gw quasiparticle energies used in
            the BSE Hamiltonian. Should match spin, k-points and
            valence/conduction bands
        truncation: str
            Coulomb truncation scheme. Can be either wigner-seitz,
            2D, 1D, or 0D
        integrate_gamma: int
            Method to integrate the Coulomb interaction. 1 is a numerical
            integration at all q-points with G=[0,0,0] - this breaks the
            symmetry slightly. 0 is analytical integration at q=[0,0,0] only -
            this conserves the symmetry. integrate_gamma=2 is the same as 1,
            but the average is only carried out in the non-periodic directions.
        txt: str
            txt output
        mode: str
            Theory level used. can be RPA TDHF or BSE. Only BSE is screened.
        wfile: str
            File for saving screened interaction and some other stuff
            needed later
        write_h: bool
            If True, write the BSE Hamiltonian to H_SS.ulm.
        write_v: bool
            If True, write eigenvalues and eigenstates to v_TS.ulm
        """

        # Calculator
        if isinstance(calc, str):
            calc = GPAW(calc, txt=None, communicator=serial_comm)
        self.calc = calc
        self.spinors = spinors
        self.scale = scale

        assert mode in ['RPA', 'TDHF', 'BSE']
        # assert calc.wfs.kd.nbzkpts % world.size == 0

        # txt file
        if world.rank != 0:
            txt = devnull
        elif isinstance(txt, str):
            txt = open(txt, 'w', 1)
        self.fd = txt

        self.ecut = ecut / Hartree
        self.nbands = nbands
        self.mode = mode
        self.truncation = truncation
        if integrate_gamma == 0 and truncation is not None:
            print('***WARNING*** Analytical Coulomb integration is ' +
                  'not expected to work with Coulomb truncation. ' +
                  'Use integrate_gamma=1', file=self.fd)
        self.integrate_gamma = integrate_gamma
        self.wfile = wfile
        self.write_h = write_h
        self.write_v = write_v

        # Find q-vectors and weights in the IBZ:
        self.kd = calc.wfs.kd
        if -1 in self.kd.bz2bz_ks:
            print('***WARNING*** Symmetries may not be right ' +
                  'Use gamma-centered grid to be sure', file=self.fd)
        offset_c = 0.5 * ((self.kd.N_c + 1) % 2) / self.kd.N_c
        bzq_qc = monkhorst_pack(self.kd.N_c) + offset_c
        self.qd = KPointDescriptor(bzq_qc)
        self.qd.set_symmetry(self.calc.atoms, self.kd.symmetry)
        self.vol = abs(np.linalg.det(calc.wfs.gd.cell_cv))

        # bands
        self.spins = self.calc.wfs.nspins
        if self.spins == 2:
            if self.spinors:
                self.spinors = False
                print('***WARNING*** Presently the spinor version' +
                      'does not work for spin-polarized calculations.' +
                      'Performing scalar calculation', file=self.fd)
            assert len(valence_bands[0]) == len(valence_bands[1])
            assert len(conduction_bands[0]) == len(conduction_bands[1])
        if valence_bands is None:
            nv = self.calc.wfs.setups.nvalence
            valence_bands = [[nv // 2 - 1]]
            if self.spins == 2:
                valence_bands *= 2
        if conduction_bands is None:
            conduction_bands = [[valence_bands[-1] + 1]]
            if self.spins == 2:
                conduction_bands *= 2

        self.val_sn = np.array(valence_bands)
        if len(np.shape(self.val_sn)) == 1:
            self.val_sn = np.array([self.val_sn])
        self.con_sn = np.array(conduction_bands)
        if len(np.shape(self.con_sn)) == 1:
            self.con_sn = np.array([self.con_sn])

        self.td = True
        for n in self.val_sn[0]:
            if n in self.con_sn[0]:
                self.td = False
        if len(self.val_sn) == 2:
            for n in self.val_sn[1]:
                if n in self.con_sn[1]:
                    self.td = False

        self.nv = len(self.val_sn[0])
        self.nc = len(self.con_sn[0])
        if eshift is not None:
            eshift /= Hartree
        if gw_skn is not None:
            assert self.nv + self.nc == len(gw_skn[0, 0])
            assert self.kd.nibzkpts == len(gw_skn[0])
            gw_skn = gw_skn[:, self.kd.bz2ibz_k]
            # assert self.kd.nbzkpts == len(gw_skn[0])
            gw_skn /= Hartree
        self.gw_skn = gw_skn
        self.eshift = eshift

        # Number of pair orbitals
        self.nS = self.kd.nbzkpts * self.nv * self.nc * self.spins
        self.nS *= (self.spinors + 1)**2

        # Wigner-Seitz stuff
        if self.truncation == 'wigner-seitz':
            self.wstc = WignerSeitzTruncatedCoulomb(self.calc.wfs.gd.cell_cv,
                                                    self.kd.N_c, self.fd)
        else:
            self.wstc = None

        self.print_initialization(self.td, self.eshift, self.gw_skn)

    def calculate(self, optical=True, ac=1.0):

        if self.spinors:
            """Calculate spinors. Here m is index of eigenvalues with SOC
            and n is the basis of eigenstates withour SOC. Below m is used
            for unoccupied states and n is used for occupied states so be
            careful!"""

            print('Diagonalizing spin-orbit Hamiltonian', file=self.fd)
            param = self.calc.parameters
            if not param['symmetry'] == 'off':
                print('Calculating KS wavefunctions without symmetry ' +
                      'for spin-orbit', file=self.fd)
                if not op.isfile('gs_nosym.gpw'):
                    calc_so = GPAW(**param)
                    calc_so.set(symmetry='off',
                                fixdensity=True,
                                txt='gs_nosym.txt')
                    calc_so.atoms = self.calc.atoms
                    calc_so.density = self.calc.density
                    calc_so.get_potential_energy()
                    calc_so.write('gs_nosym.gpw')
                calc_so = GPAW('gs_nosym.gpw', txt=None,
                               communicator=serial_comm)
                e_mk, v_knm = get_spinorbit_eigenvalues(calc_so,
                                                        return_wfs=True,
                                                        scale=self.scale)
                del calc_so
            else:
                e_mk, v_knm = get_spinorbit_eigenvalues(self.calc,
                                                        return_wfs=True,
                                                        scale=self.scale)
            e_mk /= Hartree

        # Parallelization stuff
        nK = self.kd.nbzkpts
        myKrange, myKsize, mySsize = self.parallelisation_sizes()

        # Calculate exchange interaction
        qd0 = KPointDescriptor([self.q_c])
        pd0 = PWDescriptor(self.ecut, self.calc.wfs.gd, complex, qd0)
        ikq_k = self.kd.find_k_plus_q(self.q_c)
        v_G = get_coulomb_kernel(pd0, self.kd.N_c, truncation=self.truncation,
                                 wstc=self.wstc)
        if optical:
            v_G[0] = 0.0

        self.pair = PairDensity(self.calc, self.ecut, world=serial_comm,
                                txt='pair.txt')

        # Calculate direct (screened) interaction and PAW corrections
        if self.mode == 'RPA':
            Q_aGii = self.pair.initialize_paw_corrections(pd0)
        else:
            self.get_screened_potential(ac=ac)
            if (self.qd.ibzk_kc - self.q_c < 1.0e-6).all():
                iq0 = self.qd.bz2ibz_k[self.kd.where_is_q(self.q_c,
                                                          self.qd.bzk_kc)]
                Q_aGii = self.Q_qaGii[iq0]
            else:
                Q_aGii = self.pair.initialize_paw_corrections(pd0)

        # Calculate pair densities, eigenvalues and occupations
        so = self.spinors + 1
        Nv, Nc = so * self.nv, so * self.nc
        Ns = self.spins
        rhoex_KsmnG = np.zeros((nK, Ns, Nv, Nc, len(v_G)), complex)
        # rhoG0_Ksmn = np.zeros((nK, Ns, Nv, Nc), complex)
        df_Ksmn = np.zeros((nK, Ns, Nv, Nc), float) # -(ev - ec)
        deps_ksmn = np.zeros((myKsize, Ns, Nv, Nc), float) # -(fv - fc)
        if np.allclose(self.q_c, 0.0):
            optical_limit = True
        else:
            optical_limit = False
        get_pair = self.pair.get_kpoint_pair
        get_rho = self.pair.get_pair_density
        if self.spinors:
            # Get all pair densities to allow for SOC mixing
            # Use twice as many no-SOC states as BSE bands to allow mixing
            vi_s = [2 * self.val_sn[0, 0] - self.val_sn[0, -1] - 1]
            vf_s = [2 * self.con_sn[0, -1] - self.con_sn[0, 0] + 2]
            if vi_s[0] < 0:
                vi_s[0] = 0
            ci_s, cf_s = vi_s, vf_s
            ni, nf = vi_s[0], vf_s[0]
            mvi = 2 * self.val_sn[0, 0]
            mvf = 2 * (self.val_sn[0, -1] + 1)
            mci = 2 * self.con_sn[0, 0]
            mcf = 2 * (self.con_sn[0, -1] + 1)
        else:
            vi_s, vf_s = self.val_sn[:, 0], self.val_sn[:, -1] + 1
            ci_s, cf_s = self.con_sn[:, 0], self.con_sn[:, -1] + 1
        for ik, iK in enumerate(myKrange):
            for s in range(Ns):
                pair = get_pair(pd0, s, iK,
                                vi_s[s], vf_s[s], ci_s[s], cf_s[s])
                m_m = np.arange(vi_s[s], vf_s[s])
                n_n = np.arange(ci_s[s], cf_s[s])
                if self.gw_skn is not None:
                    iKq = self.calc.wfs.kd.find_k_plus_q(self.q_c, [iK])[0]
                    epsv_m = self.gw_skn[s, iK, :self.nv]
                    epsc_n = self.gw_skn[s, iKq, self.nv:]
                    deps_ksmn[ik] = -(epsv_m[:, np.newaxis] - epsc_n)
                elif self.spinors:
                    iKq = self.calc.wfs.kd.find_k_plus_q(self.q_c, [iK])[0]
                    epsv_m = e_mk[mvi:mvf, iK]
                    epsc_n = e_mk[mci:mcf, iKq]
                    deps_ksmn[ik, s] = -(epsv_m[:, np.newaxis] - epsc_n)
                else:
                    deps_ksmn[ik, s] = -pair.get_transition_energies(m_m, n_n)

                df_mn = pair.get_occupation_differences(self.val_sn[s],
                                                        self.con_sn[s])
                rho_mnG = get_rho(pd0, pair,
                                  m_m, n_n,
                                  optical_limit=optical_limit,
                                  direction=self.direction,
                                  Q_aGii=Q_aGii,
                                  extend_head=False)
                if self.spinors:
                    if optical_limit:
                        deps0_mn = -pair.get_transition_energies(m_m, n_n)
                        rho_mnG[:, :, 0] *= deps0_mn
                    df_Ksmn[iK, s, ::2, ::2] = df_mn
                    df_Ksmn[iK, s, ::2, 1::2] = df_mn
                    df_Ksmn[iK, s, 1::2, ::2] = df_mn
                    df_Ksmn[iK, s, 1::2, 1::2] = df_mn
                    vecv0_nm = v_knm[iK][::2][ni:nf, mvi:mvf]
                    vecc0_nm = v_knm[iKq][::2][ni:nf, mci:mcf]
                    rho_0mnG = np.dot(vecv0_nm.T.conj(),
                                      np.dot(vecc0_nm.T, rho_mnG))
                    vecv1_nm = v_knm[iK][1::2][ni:nf, mvi:mvf]
                    vecc1_nm = v_knm[iKq][1::2][ni:nf, mci:mcf]
                    rho_1mnG = np.dot(vecv1_nm.T.conj(),
                                      np.dot(vecc1_nm.T, rho_mnG))
                    rhoex_KsmnG[iK, s] = rho_0mnG + rho_1mnG
                    if optical_limit:
                        rhoex_KsmnG[iK, s, :, :, 0] /= deps_ksmn[ik, s]
                else:
                    df_Ksmn[iK, s] = pair.get_occupation_differences(m_m, n_n)
                    rhoex_KsmnG[iK, s] = rho_mnG

        if self.eshift is not None:
            deps_ksmn[np.where(df_Ksmn[myKrange] > 1.0e-3)] += self.eshift
            deps_ksmn[np.where(df_Ksmn[myKrange] < -1.0e-3)] -= self.eshift

        world.sum(df_Ksmn)
        world.sum(rhoex_KsmnG)

        self.rhoG0_S = np.reshape(rhoex_KsmnG[:, :, :, :, 0], -1)

        if hasattr(self, 'H_sS'):
            return

        # Calculate Hamiltonian
        t0 = time()
        print('Calculating %s matrix elements at q_c = %s'
              % (self.mode, self.q_c), file=self.fd)
        H_ksmnKsmn = np.zeros((myKsize, Ns, Nv, Nc, nK, Ns, Nv, Nc), complex)
        for ik1, iK1 in enumerate(myKrange):
            for s1 in range(Ns):
                kptv1 = self.pair.get_k_point(s1, iK1, vi_s[s1], vf_s[s1])
                kptc1 = self.pair.get_k_point(s1, ikq_k[iK1], ci_s[s1],
                                              cf_s[s1])
                rho1_mnG = rhoex_KsmnG[iK1, s1]

                #rhoG0_Ksmn[iK1, s1] = rho1_mnG[:, :, 0]
                rho1ccV_mnG = rho1_mnG.conj()[:, :] * v_G
                for s2 in range(Ns):
                    for Q_c in self.qd.bzk_kc:
                        iK2 = self.kd.find_k_plus_q(Q_c, [kptv1.K])[0]
                        rho2_mnG = rhoex_KsmnG[iK2, s2]
                        rho2_mGn = np.swapaxes(rho2_mnG, 1, 2)
                        H_ksmnKsmn[ik1, s1, :, :, iK2, s2, :, :] += (
                            np.dot(rho1ccV_mnG, rho2_mGn))
                        if not self.mode == 'RPA' and s1 == s2:
                            ikq = ikq_k[iK2]
                            kptv2 = self.pair.get_k_point(s1, iK2, vi_s[s1],
                                                          vf_s[s1])
                            kptc2 = self.pair.get_k_point(s1, ikq, ci_s[s1],
                                                          cf_s[s1])
                            rho3_mmG, iq = self.get_density_matrix(kptv1,
                                                                   kptv2)
                            rho4_nnG, iq = self.get_density_matrix(kptc1,
                                                                   kptc2)
                            if self.spinors:
                                vec0_nm = v_knm[iK1][::2][ni:nf, mvi:mvf]
                                vec1_nm = v_knm[iK1][1::2][ni:nf, mvi:mvf]
                                vec2_nm = v_knm[iK2][::2][ni:nf, mvi:mvf]
                                vec3_nm = v_knm[iK2][1::2][ni:nf, mvi:mvf]
                                rho_0mnG = np.dot(vec0_nm.T.conj(),
                                                  np.dot(vec2_nm.T, rho3_mmG))
                                rho_1mnG = np.dot(vec1_nm.T.conj(),
                                                  np.dot(vec3_nm.T, rho3_mmG))
                                rho3_mmG = rho_0mnG + rho_1mnG
                                vec0_nm = v_knm[ikq_k[iK1]][::2][ni:nf, mci:mcf]
                                vec1_nm = v_knm[ikq_k[iK1]][1::2][ni:nf,mci:mcf]
                                vec2_nm = v_knm[ikq][::2][ni:nf, mci:mcf]
                                vec3_nm = v_knm[ikq][1::2][ni:nf, mci:mcf]
                                rho_0mnG = np.dot(vec0_nm.T.conj(),
                                                  np.dot(vec2_nm.T, rho4_nnG))
                                rho_1mnG = np.dot(vec1_nm.T.conj(),
                                                  np.dot(vec3_nm.T, rho4_nnG))
                                rho4_nnG = rho_0mnG + rho_1mnG

                            rho3ccW_mmG = np.dot(rho3_mmG.conj(),
                                                 self.W_qGG[iq])
                            W_mmnn = np.dot(rho3ccW_mmG,
                                            np.swapaxes(rho4_nnG, 1, 2))
                            W_mnmn = np.swapaxes(W_mmnn, 1, 2) * Ns * so
                            H_ksmnKsmn[ik1, s1, :, :, iK2, s1] -= 0.5 * W_mnmn
            if iK1 % (myKsize // 5 + 1) == 0:
                dt = time() - t0
                tleft = dt * myKsize / (iK1 + 1) - dt
                print('  Finished %s pair orbitals in %s - Estimated %s left' %
                      ((iK1 + 1) * Nv * Nc * Ns * world.size,
                       timedelta(seconds=round(dt)),
                       timedelta(seconds=round(tleft))), file=self.fd)

        #if self.mode == 'BSE':
        #    del self.Q_qaGii, self.W_qGG, self.pd_q

        H_ksmnKsmn /= self.vol

        mySsize = myKsize * Nv * Nc * Ns
        if myKsize > 0:
            iS0 = myKrange[0] *  Nv * Nc * Ns

        #world.sum(rhoG0_Ksmn)
        #self.rhoG0_S = np.reshape(rhoG0_Ksmn, -1)
        self.df_S = np.reshape(df_Ksmn, -1)
        if not self.td:
            self.excludef_S = np.where(np.abs(self.df_S) < 0.001)[0]
        # multiply by 2 when spin-paired and no SOC
        self.df_S *= 2.0 / nK / Ns / so
        self.deps_s = np.reshape(deps_ksmn, -1)
        H_sS = np.reshape(H_ksmnKsmn, (mySsize, self.nS))
        for iS in range(mySsize):
            # Multiply by occupations and adiabatic coupling
            H_sS[iS] *= self.df_S[iS0 + iS] * ac
            # add bare transition energies
            H_sS[iS, iS0 + iS] += self.deps_s[iS]

        self.H_sS = H_sS

        if self.write_h:
            self.par_save('H_SS.ulm', 'H_SS', self.H_sS)

    def get_density_matrix(self, kpt1, kpt2):

        Q_c = self.kd.bzk_kc[kpt2.K] - self.kd.bzk_kc[kpt1.K]
        iQ = self.qd.where_is_q(Q_c, self.qd.bzk_kc)
        iq = self.qd.bz2ibz_k[iQ]
        q_c = self.qd.ibzk_kc[iq]

        # Find symmetry that transforms Q_c into q_c
        sym = self.qd.sym_k[iQ]
        U_cc = self.qd.symmetry.op_scc[sym]
        time_reversal = self.qd.time_reversal_k[iQ]
        sign = 1 - 2 * time_reversal
        d_c = sign * np.dot(U_cc, q_c) - Q_c
        assert np.allclose(d_c.round(), d_c)

        pd = self.pd_q[iq]
        N_c = pd.gd.N_c
        i_cG = sign * np.dot(U_cc, np.unravel_index(pd.Q_qG[0], N_c))

        shift0_c = Q_c - sign * np.dot(U_cc, q_c)
        assert np.allclose(shift0_c.round(), shift0_c)
        shift0_c = shift0_c.round().astype(int)

        shift_c = kpt1.shift_c - kpt2.shift_c - shift0_c
        I_G = np.ravel_multi_index(i_cG + shift_c[:, None], N_c, 'wrap')
        G_Gv = pd.get_reciprocal_vectors()
        pos_ac = self.calc.spos_ac
        pos_av = np.dot(pos_ac, pd.gd.cell_cv)
        M_vv = np.dot(pd.gd.cell_cv.T, np.dot(U_cc.T,
                                              np.linalg.inv(pd.gd.cell_cv).T))

        Q_aGii = []
        for a, Q_Gii in enumerate(self.Q_qaGii[iq]):
            x_G = np.exp(1j * np.dot(G_Gv, (pos_av[a] -
                                            np.dot(M_vv, pos_av[a]))))
            U_ii = self.calc.wfs.setups[a].R_sii[sym]
            Q_Gii = np.dot(np.dot(U_ii, Q_Gii * x_G[:, None, None]),
                           U_ii.T).transpose(1, 0, 2)
            if sign == -1:
                Q_Gii = Q_Gii.conj()
            Q_aGii.append(Q_Gii)

        rho_mnG = np.zeros((len(kpt1.eps_n), len(kpt2.eps_n), len(G_Gv)),
                           complex)
        for m in range(len(rho_mnG)):
            C1_aGi = [np.dot(Qa_Gii, P1_ni[m].conj())
                      for Qa_Gii, P1_ni in zip(Q_aGii, kpt1.P_ani)]
            ut1cc_R = kpt1.ut_nR[m].conj()
            rho_mnG[m] = self.pair.calculate_pair_densities(ut1cc_R, C1_aGi,
                                                            kpt2, pd, I_G)
        return rho_mnG, iq

    def get_screened_potential(self, ac=1.0):

        if hasattr(self, 'W_qGG'):
            return

        if self.wfile is not None:
            # Read screened potential from file
            try:
                data = np.load(self.wfile + '.npz')
                self.Q_qaGii = data['Q']
                self.W_qGG = data['W']
                self.pd_q = data['pd']
                print('Reading screened potential from % s' % self.wfile,
                      file=self.fd)
            except:
                self.calculate_screened_potential(ac)
                print('Saving screened potential to % s' % self.wfile,
                      file=self.fd)
                if world.rank == 0:
                    np.savez(self.wfile,
                             Q=self.Q_qaGii, pd=self.pd_q, W=self.W_qGG)
        else:
            self.calculate_screened_potential(ac)

    def calculate_screened_potential(self, ac):
        """Calculate W_GG(q)"""

        chi0 = Chi0(self.calc,
                    frequencies=[0.0],
                    eta=0.001,
                    ecut=self.ecut,
                    intraband=False,
                    hilbert=False,
                    nbands=self.nbands,
                    txt='chi0.txt',
                    world=world,
                    )

        self.blockcomm = chi0.blockcomm
        wfs = self.calc.wfs

        self.Q_qaGii = []
        self.W_qGG = []
        self.pd_q = []

        t0 = time()
        print('Calculating screened potential', file=self.fd)
        for iq, q_c in enumerate(self.qd.ibzk_kc):
            thisqd = KPointDescriptor([q_c])
            pd = PWDescriptor(self.ecut, wfs.gd, complex, thisqd)
            nG = pd.ngmax

            chi0.Ga = self.blockcomm.rank * nG
            chi0.Gb = min(chi0.Ga + nG, nG)
            chi0_wGG = np.zeros((1, nG, nG), complex)
            if np.allclose(q_c, 0.0):
                chi0_wxvG = np.zeros((1, 2, 3, nG), complex)
                chi0_wvv = np.zeros((1, 3, 3), complex)
            else:
                chi0_wxvG = None
                chi0_wvv = None

            chi0._calculate(pd, chi0_wGG, chi0_wxvG, chi0_wvv,
                            0, self.nbands, spins='all', extend_head=False)
            chi0_GG = chi0_wGG[0]

            # Calculate eps^{-1}_GG
            if pd.kd.gamma:
                # Generate fine grid in vicinity of gamma
                kd = self.calc.wfs.kd
                N = 4
                N_c = np.array([N, N, N])
                if self.truncation is not None:
                    # Only average periodic directions if trunction is used
                    N_c[kd.N_c == 1] = 1
                qf_qc = monkhorst_pack(N_c) / kd.N_c
                qf_qc *= 1.0e-6
                U_scc = kd.symmetry.op_scc
                qf_qc = kd.get_ibz_q_points(qf_qc, U_scc)[0]
                weight_q = kd.q_weights
                qf_qv = 2 * np.pi * np.dot(qf_qc, pd.gd.icell_cv)
                a_q = np.sum(np.dot(chi0_wvv[0], qf_qv.T) * qf_qv.T, axis=0)
                a0_qG = np.dot(qf_qv, chi0_wxvG[0, 0])
                a1_qG = np.dot(qf_qv, chi0_wxvG[0, 1])
                einv_GG = np.zeros((nG, nG), complex)
                # W_GG = np.zeros((nG, nG), complex)
                for iqf in range(len(qf_qv)):
                    chi0_GG[0] = a0_qG[iqf]
                    chi0_GG[:, 0] = a1_qG[iqf]
                    chi0_GG[0, 0] = a_q[iqf]
                    sqrV_G = get_coulomb_kernel(pd,
                                                kd.N_c,
                                                truncation=self.truncation,
                                                wstc=self.wstc,
                                                q_v=qf_qv[iqf])**0.5
                    sqrV_G *= ac**0.5  # Multiply by adiabatic coupling
                    e_GG = np.eye(nG) - chi0_GG * sqrV_G * sqrV_G[:,
                                                                  np.newaxis]
                    einv_GG += np.linalg.inv(e_GG) * weight_q[iqf]
                    # einv_GG = np.linalg.inv(e_GG) * weight_q[iqf]
                    # W_GG += (einv_GG * sqrV_G * sqrV_G[:, np.newaxis]
                    #          * weight_q[iqf])
            else:
                sqrV_G = get_coulomb_kernel(pd,
                                            self.kd.N_c,
                                            truncation=self.truncation,
                                            wstc=self.wstc)**0.5
                sqrV_G *= ac**0.5  # Multiply by adiabatic coupling
                e_GG = np.eye(nG) - chi0_GG * sqrV_G * sqrV_G[:, np.newaxis]
                einv_GG = np.linalg.inv(e_GG)
                # W_GG = einv_GG * sqrV_G * sqrV_G[:, np.newaxis]

            # Now calculate W_GG
            if pd.kd.gamma:
                # Reset bare Coulomb interaction
                sqrV_G = get_coulomb_kernel(pd,
                                            self.kd.N_c,
                                            truncation=self.truncation,
                                            wstc=self.wstc)**0.5
            W_GG = einv_GG * sqrV_G * sqrV_G[:, np.newaxis]
            if self.integrate_gamma != 0:
                # Numerical integration of Coulomb interaction at all q-points
                if self.integrate_gamma == 2:
                    reduced = True
                else:
                    reduced = False
                V0, sqrV0 = get_integrated_kernel(pd,
                                                  self.kd.N_c,
                                                  truncation=self.truncation,
                                                  reduced=reduced,
                                                  N=100)
                W_GG[0, 0] = einv_GG[0, 0] * V0
                W_GG[0, 1:] = einv_GG[0, 1:] * sqrV0 * sqrV_G[1:]
                W_GG[1:, 0] = einv_GG[1:, 0] * sqrV_G[1:] * sqrV0
            elif self.integrate_gamma == 0 and pd.kd.gamma:
                # Analytical integration at gamma
                bzvol = (2 * np.pi)**3 / self.vol / self.qd.nbzkpts
                Rq0 = (3 * bzvol / (4 * np.pi))**(1. / 3.)
                V0 = 16 * np.pi**2 * Rq0 / bzvol
                sqrV0 = (4 * np.pi)**(1.5) * Rq0**2 / bzvol / 2
                W_GG[0, 0] = einv_GG[0, 0] * V0
                W_GG[0, 1:] = einv_GG[0, 1:] * sqrV0 * sqrV_G[1:]
                W_GG[1:, 0] = einv_GG[1:, 0] * sqrV_G[1:] * sqrV0
            else:
                pass

            if pd.kd.gamma:
                e = 1 / einv_GG[0, 0].real
                print('    RPA dielectric constant is: %3.3f' % e,
                      file=self.fd)
            self.Q_qaGii.append(chi0.Q_aGii)
            self.pd_q.append(pd)
            self.W_qGG.append(W_GG)

            if iq % (self.qd.nibzkpts // 5 + 1) == 2:
                dt = time() - t0
                tleft = dt * self.qd.nibzkpts / (iq + 1) - dt
                print('  Finished %s q-points in %s - Estimated %s left' %
                      (iq + 1, timedelta(seconds=round(dt)),
                       timedelta(seconds=round(tleft))), file=self.fd)

    def diagonalize(self):

        print('Diagonalizing Hamiltonian', file=self.fd)
        """The t and T represent local and global
           eigenstates indices respectively
        """

        # Non-Hermitian matrix can only use linalg.eig
        if not self.td:
            print('  Using numpy.linalg.eig...', file=self.fd)
            print('  Eliminated %s pair orbitals' % len(self.excludef_S),
                  file=self.fd)

            self.H_SS = self.collect_A_SS(self.H_sS)
            self.w_T = np.zeros(self.nS - len(self.excludef_S), complex)
            if world.rank == 0:
                self.H_SS = np.delete(self.H_SS, self.excludef_S, axis=0)
                self.H_SS = np.delete(self.H_SS, self.excludef_S, axis=1)
                self.w_T, self.v_ST = np.linalg.eig(self.H_SS)
            world.broadcast(self.w_T, 0)
            self.df_S = np.delete(self.df_S, self.excludef_S)
            self.rhoG0_S = np.delete(self.rhoG0_S, self.excludef_S)
        # Here the eigenvectors are returned as complex conjugated rows
        else:
            if world.size == 1:
                print('  Using lapack...', file=self.fd)
                from gpaw.utilities.lapack import diagonalize
                self.w_T = np.zeros(self.nS)
                diagonalize(self.H_sS, self.w_T)
                self.v_St = self.H_sS.conj().T
            else:
                print('  Using scalapack...', file=self.fd)
                nS = self.nS
                ns = -(-self.kd.nbzkpts // world.size) * (self.nv * self.nc *
                                                          self.spins *
                                                          (self.spinors + 1)**2)
                grid = BlacsGrid(world, world.size, 1)
                desc = grid.new_descriptor(nS, nS, ns, nS)

                desc2 = grid.new_descriptor(nS, nS, 2, 2)
                H_tmp = desc2.zeros(dtype=complex)
                r = Redistributor(world, desc, desc2)
                r.redistribute(self.H_sS, H_tmp)

                self.w_T = np.empty(nS)
                v_tmp = desc2.empty(dtype=complex)
                desc2.diagonalize_dc(H_tmp, v_tmp, self.w_T)

                r = Redistributor(grid.comm, desc2, desc)
                self.v_St = desc.zeros(dtype=complex)
                r.redistribute(v_tmp, self.v_St)
                self.v_St = self.v_St.conj().T

        if self.write_v and self.td:
            # Cannot use par_save without td
            self.par_save('v_TS.ulm', 'v_TS', self.v_St.T)

        return

    def get_bse_matrix(self, q_c=[0.0, 0.0, 0.0], direction=0, ac=1.0,
                       readfile=None, optical=True, write_eig=None):
        """Calculate and diagonalize BSE matrix"""

        self.q_c = q_c
        self.direction = direction

        if readfile is None:
            self.calculate(optical=optical, ac=ac)
            if hasattr(self, 'w_T'):
                return
            self.diagonalize()
        elif readfile == 'H_SS':
            print('Reading Hamiltonian from file', file=self.fd)
            self.par_load('H_SS.ulm', 'H_SS')
            self.diagonalize()
        elif readfile == 'v_TS':
            print('Reading eigenstates from file', file=self.fd)
            self.par_load('v_TS.ulm', 'v_TS')
        else:
            raise ValueError('%s array not recognized' % readfile)

        # TODO: Move write_eig here

        return

    def get_vchi(self, w_w=None, eta=0.1, q_c=[0.0, 0.0, 0.0],
                 direction=0, ac=1.0, readfile=None, optical=True,
                 write_eig=None):
        """Returns v * \chi where v is the bare Coulomb interaction"""

        self.get_bse_matrix(q_c=q_c, direction=direction, ac=ac,
                            readfile=readfile, optical=optical,
                            write_eig=write_eig)

        w_T = self.w_T
        rhoG0_S = self.rhoG0_S
        df_S = self.df_S

        print('Calculating response function at %s frequency points' %
              len(w_w), file=self.fd)
        vchi_w = np.zeros(len(w_w), dtype=complex)

        if not self.td:
            C_T = np.zeros(self.nS - len(self.excludef_S), complex)
            if world.rank == 0:
                A_T = np.dot(rhoG0_S, self.v_ST)
                B_T = np.dot(rhoG0_S * df_S, self.v_ST)
                tmp = np.dot(self.v_ST.conj().T, self.v_ST)
                overlap_tt = np.linalg.inv(tmp)
                C_T = np.dot(B_T.conj(), overlap_tt.T) * A_T
            world.broadcast(C_T, 0)
        else:
            A_t = np.dot(rhoG0_S, self.v_St)
            B_t = np.dot(rhoG0_S * df_S, self.v_St)
            if world.size == 1:
                C_T = B_t.conj() * A_t
            else:
                Nv = self.nv * (self.spinors + 1)
                Nc = self.nc * (self.spinors + 1)
                Ns = self.spins
                nS = self.nS
                ns = -(-self.kd.nbzkpts // world.size) * Nv * Nc * Ns
                grid = BlacsGrid(world, world.size, 1)
                desc = grid.new_descriptor(nS, 1, ns, 1)
                C_t = desc.empty(dtype=complex)
                C_t[:, 0] = B_t.conj() * A_t
                C_T = desc.collect_on_master(C_t)[:, 0]
                if world.rank != 0:
                    C_T = np.empty(nS, dtype=complex)
                world.broadcast(C_T, 0)

        eta /= Hartree
        for iw, w in enumerate(w_w / Hartree):
            tmp_T = 1. / (w - w_T + 1j * eta)
            vchi_w[iw] += np.dot(tmp_T, C_T)
        vchi_w *= 4 * np.pi / self.vol

        if not np.allclose(self.q_c, 0.0):
            cell_cv = self.calc.wfs.gd.cell_cv
            B_cv = 2 * np.pi * np.linalg.inv(cell_cv).T
            q_v = np.dot(q_c, B_cv)
            vchi_w /= np.dot(q_v, q_v)

        """Check f-sum rule."""
        nv = self.calc.wfs.setups.nvalence
        dw_w = (w_w[1:] - w_w[:-1]) / Hartree
        wchi_w = (w_w[1:] * vchi_w[1:] + w_w[:-1] * vchi_w[:-1]) / Hartree / 2
        N = -np.dot(dw_w, wchi_w.imag) * self.vol / (2 * np.pi**2)
        print(file=self.fd)
        print('Checking f-sum rule:', file=self.fd)
        print('  Valence = %s, N = %f' % (nv, N), file=self.fd)
        print(file=self.fd)

        if write_eig is not None:
            if world.rank == 0:
                f = open(write_eig, 'w')
                print('# %s eigenvalues in eV' % self.mode, file=f)
                for iw, w in enumerate(self.w_T * Hartree):
                    print('%8d %12.6f %12.16f' % (iw, w.real, C_T[iw].real),
                          file=f)
                f.close()

        return vchi_w * ac

    def get_dielectric_function(self, w_w=None, eta=0.1,
                                q_c=[0.0, 0.0, 0.0], direction=0,
                                filename='df_bse.csv', readfile=None,
                                write_eig='eig.dat'):
        """Returns and writes real and imaginary part of the dielectric
        function.

        w_w: list of frequencies (eV)
            Dielectric function is calculated at these frequencies
        eta: float
            Lorentzian broadening of the spectrum (eV)
        q_c: list of three floats
            Wavevector in reduced units on which the response is calculated
        direction: int
            if q_c = [0, 0, 0] this gives the direction in cartesian
            coordinates - 0=x, 1=y, 2=z
        filename: str
            data file on which frequencies, real and imaginary part of
            dielectric function is written
        readfile: str
            If H_SS is given, the method will load the BSE Hamiltonian
            from H_SS.ulm. If v_TS is given, the method will load the
            eigenstates from v_TS.ulm
        write_eig: str
            File on which the BSE eigenvalues are written
        """

        epsilon_w = -self.get_vchi(w_w=w_w, eta=eta, q_c=q_c,
                                   direction=direction,
                                   readfile=readfile, optical=True,
                                   write_eig=write_eig)
        epsilon_w += 1.0

        if world.rank == 0 and filename is not None:
            f = open(filename, 'w')
            for iw, w in enumerate(w_w):
                print('%.9f, %.9f, %.9f' %
                      (w, epsilon_w[iw].real, epsilon_w[iw].imag), file=f)
            f.close()
        world.barrier()

        print('Calculation completed at:', ctime(), file=self.fd)
        print(file=self.fd)

        return w_w, epsilon_w

    def get_eels_spectrum(self, w_w=None, eta=0.1,
                          q_c=[0.0, 0.0, 0.0], direction=0,
                          filename='df_bse.csv', readfile=None,
                          write_eig='eig.dat'):
        """Returns and writes real and imaginary part of the dielectric
        function.

        w_w: list of frequencies (eV)
            Dielectric function is calculated at these frequencies
        eta: float
            Lorentzian broadening of the spectrum (eV)
        q_c: list of three floats
            Wavevector in reduced units on which the response is calculated
        direction: int
            if q_c = [0, 0, 0] this gives the direction in cartesian
            coordinates - 0=x, 1=y, 2=z
        filename: str
            data file on which frequencies, real and imaginary part of
            dielectric function is written
        readfile: str
            If H_SS is given, the method will load the BSE Hamiltonian
            from H_SS.ulm. If v_TS is given, the method will load the
            eigenstates from v_TS.ulm
        write_eig: str
            File on which the BSE eigenvalues are written
        """

        eels_w = -self.get_vchi(w_w=w_w, eta=eta, q_c=q_c, direction=direction,
                                readfile=readfile, optical=False,
                                write_eig=write_eig).imag

        if world.rank == 0 and filename is not None:
            f = open(filename, 'w')
            for iw, w in enumerate(w_w):
                print('%.9f, %.9f' % (w, eels_w[iw]), file=f)
            f.close()
        world.barrier()

        print('Calculation completed at:', ctime(), file=self.fd)
        print(file=self.fd)

        return w_w, eels_w

    def get_polarizability(self, w_w=None, eta=0.1,
                           q_c=[0.0, 0.0, 0.0], direction=0,
                           filename='pol_bse.csv', readfile=None, pbc=None,
                           write_eig='eig.dat'):
        """Calculate the polarizability alpha.
        In 3D the imaginary part of the polarizability is related to the
        dielectric function by Im(eps_M) = 4 pi * Im(alpha). In systems
        with reduced dimensionality the converged value of alpha is
        independent of the cell volume. This is not the case for eps_M,
        which is ill defined. A truncated Coulomb kernel will always give
        eps_M = 1.0, whereas the polarizability maintains its structure.
        pbs should be a list of booleans giving the periodic directions.

        By default, generate a file 'pol_bse.csv'. The three colomns are:
        frequency (eV), Real(alpha), Imag(alpha). The dimension of alpha
        is \AA to the power of non-periodic directions.
        """

        cell_cv = self.calc.wfs.gd.cell_cv
        if not pbc:
            pbc_c = self.calc.atoms.pbc
        else:
            pbc_c = np.array(pbc)
        if pbc_c.all():
            V = 1.0
        else:
            V = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))

        if self.truncation is None:
            optical = True
        else:
            optical = False

        vchi_w = self.get_vchi(w_w=w_w, eta=eta, q_c=q_c, direction=direction,
                               readfile=readfile, optical=optical,
                               write_eig=write_eig)
        alpha_w = -V * vchi_w / (4 * np.pi)
        alpha_w *= Bohr**(sum(~pbc_c))

        if world.rank == 0 and filename is not None:
            fd = open(filename, 'w')
            for iw, w in enumerate(w_w):
                print('%.9f, %.9f, %.9f' %
                      (w, alpha_w[iw].real, alpha_w[iw].imag), file=fd)
            fd.close()

        print('Calculation completed at:', ctime(), file=self.fd)
        print(file=self.fd)

        return w_w, alpha_w

    def get_2d_absorption(self, w_w=None, eta=0.1,
                          q_c=[0.0, 0.0, 0.0], direction=0,
                          filename='abs_bse.csv', readfile=None, pbc=None,
                          write_eig='eig.dat'):
        """Calculate the dimensionless absorption for 2d materials.
        It is essentially related to the 2D polarizability \alpha_2d as

              ABS = 4 * np.pi * \omega * \alpha_2d / c

        where c is the velocity of light
        """

        from ase.units import alpha
        c = 1.0 / alpha

        cell_cv = self.calc.wfs.gd.cell_cv
        if not pbc:
            pbc_c = self.calc.atoms.pbc
        else:
            pbc_c = np.array(pbc)

        assert np.sum(pbc_c) == 2
        V = np.abs(np.linalg.det(cell_cv[~pbc_c][:, ~pbc_c]))
        vchi_w = self.get_vchi(w_w=w_w, eta=eta, q_c=q_c, direction=direction,
                               readfile=readfile, optical=True,
                               write_eig=write_eig)
        abs_w = -V * vchi_w.imag * w_w / Hartree / c

        if world.rank == 0 and filename is not None:
            fd = open(filename, 'w')
            for iw, w in enumerate(w_w):
                print('%.9f, %.9f' % (w, abs_w[iw]), file=fd)
            fd.close()

        print('Calculation completed at:', ctime(), file=self.fd)
        print(file=self.fd)

        return w_w, abs_w

    def par_save(self, filename, name, A_sS):
        import ase.io.ulm as ulm

        if world.size == 1:
            A_XS = A_sS
        else:
            A_XS = self.collect_A_SS(A_sS)

        if world.rank == 0:
            w = ulm.open(filename, 'w')
            if name == 'v_TS':
                w.write(w_T=self.w_T)
            # w.write(nS=self.nS)
            w.write(rhoG0_S=self.rhoG0_S)
            w.write(df_S=self.df_S)
            w.write(A_XS=A_XS)
            w.close()
        world.barrier()

    def par_load(self, filename, name):
        import ase.io.ulm as ulm

        if world.rank == 0:
            r = ulm.open(filename, 'r')
            if name == 'v_TS':
                self.w_T = r.w_T
            self.rhoG0_S = r.rhoG0_S
            self.df_S = r.df_S
            A_XS = r.A_XS
            r.close()
        else:
            if name == 'v_TS':
                self.w_T = np.zeros((self.nS), dtype=float)
            self.rhoG0_S = np.zeros((self.nS), dtype=complex)
            self.df_S = np.zeros((self.nS), dtype=float)
            A_XS = None

        world.broadcast(self.rhoG0_S, 0)
        world.broadcast(self.df_S, 0)

        if name == 'H_SS':
            self.H_sS = self.distribute_A_SS(A_XS)

        if name == 'v_TS':
            world.broadcast(self.w_T, 0)
            self.v_St = self.distribute_A_SS(A_XS, transpose=True)

    def collect_A_SS(self, A_sS):
        if world.rank == 0:
            A_SS = np.zeros((self.nS, self.nS), dtype=complex)
            A_SS[:len(A_sS)] = A_sS
            Ntot = len(A_sS)
            for rank in range(1, world.size):
                nkr, nk, ns = self.parallelisation_sizes(rank)
                buf = np.empty((ns, self.nS), dtype=complex)
                world.receive(buf, rank, tag=123)
                A_SS[Ntot:Ntot + ns] = buf
                Ntot += ns
        else:
            world.send(A_sS, 0, tag=123)
        world.barrier()
        if world.rank == 0:
            return A_SS

    def distribute_A_SS(self, A_SS, transpose=False):
        if world.rank == 0:
            for rank in range(0, world.size):
                nkr, nk, ns = self.parallelisation_sizes(rank)
                if rank == 0:
                    A_sS = A_SS[0:ns]
                    Ntot = ns
                else:
                    world.send(A_SS[Ntot:Ntot + ns], rank, tag=123)
                    Ntot += ns
        else:
            nkr, nk, ns = self.parallelisation_sizes()
            A_sS = np.empty((ns, self.nS), dtype=complex)
            world.receive(A_sS, 0, tag=123)
        world.barrier()
        if transpose:
            A_sS = A_sS.T
        return A_sS

    def parallelisation_sizes(self, rank=None):
        if rank is None:
            rank = world.rank
        nK = self.kd.nbzkpts
        myKsize = -(-nK // world.size)
        myKrange = range(rank * myKsize,
                         min((rank + 1) * myKsize, nK))
        myKsize = len(myKrange)
        mySsize = myKsize * self.nv * self.nc * self.spins
        mySsize *= (1 + self.spinors)**2
        return myKrange, myKsize, mySsize

    def get_bse_wf(self):
        pass
        # asd = 1.0

    def print_initialization(self, td, eshift, gw_skn):
        p = functools.partial(print, file=self.fd)
        p('----------------------------------------------------------')
        p('%s Hamiltonian' % self.mode)
        p('----------------------------------------------------------')
        p('Started at:  ', ctime())
        p()
        p('Atoms                          :',
          self.calc.atoms.get_chemical_formula(mode='hill'))
        p('Ground state XC functional     :', self.calc.hamiltonian.xc.name)
        p('Valence electrons              :', self.calc.wfs.setups.nvalence)
        p('Spinor calculations            :', self.spinors)
        p('Number of bands                :', self.calc.wfs.bd.nbands)
        p('Number of spins                :', self.calc.wfs.nspins)
        p('Number of k-points             :', self.kd.nbzkpts)
        p('Number of irreducible k-points :', self.kd.nibzkpts)
        p('Number of q-points             :', self.qd.nbzkpts)
        p('Number of irreducible q-points :', self.qd.nibzkpts)
        p()
        for q in self.qd.ibzk_kc:
            p('    q: [%1.4f %1.4f %1.4f]' % (q[0], q[1], q[2]))
        p()
        if gw_skn is not None:
            p('User specified BSE bands')
        p('Response PW cutoff             :', self.ecut * Hartree, 'eV')
        p('Screening bands included       :', self.nbands)
        if len(self.val_sn) == 1:
            p('Valence bands                  :', self.val_sn[0])
            p('Conduction bands               :', self.con_sn[0])
        else:
            p('Valence bands                  :', self.val_sn[0],self.val_sn[1])
            p('Conduction bands               :', self.con_sn[0],self.con_sn[1])
        if eshift is not None:
            p('Scissors operator              :', eshift * Hartree, 'eV')
        p('Tamm-Dancoff approximation     :', td)
        p('Number of pair orbitals        :', self.nS)
        p()
        p('Truncation of Coulomb kernel   :', self.truncation)
        if self.integrate_gamma == 0:
            p('Coulomb integration scheme     :', 'Analytical - gamma only')
        elif self.integrate_gamma == 1:
            p('Coulomb integration scheme     :', 'Numerical - all q-points')
        else:
            pass
        p()
        p('----------------------------------------------------------')
        p('----------------------------------------------------------')
        p()
        p('Parallelization - Total number of CPUs   : % s' % world.size)
        p('  Screened potential')
        p('    K-point/band decomposition           : % s' % world.size)
        p('  Hamiltonian')
        p('    Pair orbital decomposition           : % s' % world.size)
        p()
