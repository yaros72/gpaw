from __future__ import division, print_function

import sys
from math import pi
import pickle

import numpy as np

from ase.utils.timing import timer
from ase.units import Hartree
from ase.dft.kpoints import monkhorst_pack

import gpaw.mpi as mpi
from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.response.chi0 import HilbertTransform, frequency_grid
from gpaw.response.pair import PairDensity
from gpaw.wavefunctions.pw import PWDescriptor
from gpaw.xc.exx import select_kpts


class GWQEHCorrection(PairDensity):
    def __init__(self, calc, gwfile, filename=None, kpts=[0], bands=None,
                 structure=None, d=None, layer=0, 
                 dW_qw=None, qqeh=None, wqeh=None,
                 txt=sys.stdout, world=mpi.world, domega0=0.025, 
                 omega2=10.0, eta=0.1, include_q0=True, metal=False): 
        """ 
        Class for calculating quasiparticle energies of van der Waals
        heterostructures using the GW approximation for the self-energy. 
        The quasiparticle energy correction due to increased screening from
        surrounding layers is obtained from the QEH model.
        Parameters:
        
        calc: str or PAW object
            GPAW calculator object or filename of saved calculator object.
        gwfile: str
            name of gw results file from the monolayer calculation
        filename: str
            filename for gwqeh output
        kpts: list
            List of indices of sthe IBZ k-points to calculate the quasi
            particle energies for. Set to [0] by default since the QP 
            correction is generally the same for all k. 
        bands: tuple
            Range of band indices, like (n1, n2+1), to calculate the quasi
            particle energies for. Note that the second band index is not
            included. Should be the same as used for the GW calculation.
        structure: list of str
            Heterostructure set up. Each entry should consist of number of
            layers + chemical formula.
            For example: ['3H-MoS2', graphene', '10H-WS2'] gives 3 layers of
            H-MoS2, 1 layer of graphene and 10 layers of H-WS2.
            The name of the layers should correspond to building block files:
            "<name>-chi.npz" in the local repository.
        d: array of floats
            Interlayer distances for neighboring layers in Ang.
            Length of array = number of layers - 1
        layer: int
            index of layer to calculate QP correction for. 
        dW_qw: 2D array of floats dimension q X w
            Change in screened interaction. Should be set to None to calculate 
            dW directly from buildingblocks.
        qqeh: array of floats
            q-grid used for dW_qw (only needed if dW is given by hand).
        wqeh: array of floats
            w-grid used for dW_qw. So far this have to be the same as for the 
            GWQEH calculation.  (only needed if dW is given by hand).
        domega0: float
            Minimum frequency step (in eV) used in the generation of the non-
            linear frequency grid.
        omega2: float
            Control parameter for the non-linear frequency grid, equal to the
            frequency where the grid spacing has doubled in size.
        eta: float
            Broadening parameter.
        include_q0: bool
            include q=0 in W or not. if True an integral arround q=0 is
            performed, if False the q=0 contribution is set to zero. 
        metal: bool
            If True, the point at q=0 is omitted when averaging the screened
            potential close to q=0. 
        """
        
        self.gwfile = gwfile

        self.inputcalc = calc
        # Set low ecut in order to use PairDensity object since only
        # G=0 is needed.
        self.ecut = 1.
        PairDensity.__init__(self, calc, ecut=self.ecut, world=world,
                             txt=filename + '.txt')    
    
        self.filename = filename
        self.ecut /= Hartree
        self.eta = eta / Hartree
        self.domega0 = domega0 / Hartree
        self.omega2 = omega2 / Hartree

        self.kpts = list(select_kpts(kpts, self.calc))

        if bands is None:
            bands = [0, self.nocc2]
            
        self.bands = bands

        b1, b2 = bands
        self.shape = shape = (self.calc.wfs.nspins, len(self.kpts), b2 - b1)
        self.eps_sin = np.empty(shape)     # KS-eigenvalues
        self.f_sin = np.empty(shape)       # occupation numbers
        self.sigma_sin = np.zeros(shape)   # self-energies
        self.dsigma_sin = np.zeros(shape)  # derivatives of self-energies
        self.Z_sin = None                  # renormalization factors
        self.qp_sin = None
        self.Qp_sin = None
        
        self.ecutnb = 150 / Hartree
        vol = abs(np.linalg.det(self.calc.wfs.gd.cell_cv))
        self.vol = vol
        self.nbands = min(self.calc.get_number_of_bands(),
                          int(vol * (self.ecutnb)**1.5 * 2**0.5 / 3 / pi**2))

        self.nspins = self.calc.wfs.nspins

        kd = self.calc.wfs.kd

        self.mysKn1n2 = None  # my (s, K, n1, n2) indices
        self.distribute_k_points_and_bands(b1, b2, kd.ibz2bz_k[self.kpts])
        
        # Find q-vectors and weights in the IBZ:
        assert -1 not in kd.bz2bz_ks
        offset_c = 0.5 * ((kd.N_c + 1) % 2) / kd.N_c
        bzq_qc = monkhorst_pack(kd.N_c) + offset_c
        self.qd = KPointDescriptor(bzq_qc)
        self.qd.set_symmetry(self.calc.atoms, kd.symmetry)

        # frequency grid
        omax = self.find_maximum_frequency()
        self.omega_w = frequency_grid(self.domega0, self.omega2, omax)
        self.nw = len(self.omega_w)
        self.wsize = 2 * self.nw

        # Calculate screened potential of Heterostructure
        if dW_qw is None:
            try:
                data = np.load(filename + "_dW_qw.npz")
                self.qqeh = data['qqeh']
                self.wqeh = data['wqeh']
                dW_qw = data['dW_qw']
            except IOError:
                dW_qw = self.calculate_W_QEH(structure, d, layer)
        else:
            self.qqeh = qqeh
            self.wqeh = None  # wqeh
        
        self.dW_qw = self.get_W_on_grid(dW_qw, include_q0=include_q0,
                                        metal=metal)

        assert self.nw == self.dW_qw.shape[1], \
            ('Frequency grids doesnt match!')

        self.htp = HilbertTransform(self.omega_w, self.eta, gw=True)
        self.htm = HilbertTransform(self.omega_w, -self.eta, gw=True)

        self.complete = False
        self.nq = 0
        if self.load_state_file():
            if self.complete:
                print('Self-energy loaded from file', file=self.fd)
        
    def calculate_QEH(self):
        print('Calculating QEH self-energy contribution', file=self.fd)

        kd = self.calc.wfs.kd

        # Reset calculation
        self.sigma_sin = np.zeros(self.shape)   # self-energies
        self.dsigma_sin = np.zeros(self.shape)  # derivatives of self-energies

        # Get KS eigenvalues and occupation numbers:
        b1, b2 = self.bands
        nibzk = self.calc.wfs.kd.nibzkpts
        for i, k in enumerate(self.kpts):
            for s in range(self.nspins):
                u = s * nibzk + k
                kpt = self.calc.wfs.kpt_u[u]
                self.eps_sin[s, i] = kpt.eps_n[b1:b2]
                self.f_sin[s, i] = kpt.f_n[b1:b2] / kpt.weight

        # My part of the states we want to calculate QP-energies for:
        mykpts = [self.get_k_point(s, K, n1, n2)
                  for s, K, n1, n2 in self.mysKn1n2]

        Nq = len((self.qd.ibzk_kc))
        for iq, q_c in enumerate(self.qd.ibzk_kc):
            self.nq = iq
            nq = iq
            self.save_state_file()
     
            qcstr = '(' + ', '.join(['%.3f' % x for x in q_c]) + ')'
            print('Calculating contribution from IBZ q-point #%d/%d q_c=%s'
                  % (nq, Nq, qcstr), file=self.fd)
            

            # Screened potential
            dW_w = self.dW_qw[nq]
            dW_w = dW_w[:, np.newaxis, np.newaxis]
            L = abs(self.calc.wfs.gd.cell_cv[2, 2])
            dW_w *= L 

            nw = self.nw

            Wpm_w = np.zeros([2 * nw, 1, 1], dtype=complex)
            Wpm_w[:nw] = dW_w
            Wpm_w[nw:] = Wpm_w[0:nw]
            
            with self.timer('Hilbert transform'):
                self.htp(Wpm_w[:nw])
                self.htm(Wpm_w[nw:])

            qd = KPointDescriptor([q_c])
            pd0 = PWDescriptor(self.ecut, self.calc.wfs.gd, complex, qd)

            # modify pd0 by hand - only G=0 component is needed
            pd0.G_Qv = np.array([1e-17, 1e-17, 1e-17])[np.newaxis, :] 
            pd0.Q_qG = [np.array([0], dtype='int32')]
            pd0.ngmax = 1
            G_Gv = pd0.get_reciprocal_vectors()
        
            self.Q_aGii = self.initialize_paw_corrections(pd0)

            # Loop over all k-points in the BZ and find those that are related
            # to the current IBZ k-point by symmetry
            Q1 = self.qd.ibz2bz_k[iq]
            Q2s = set()
            for s, Q2 in enumerate(self.qd.bz2bz_ks[Q1]):
                if Q2 >= 0 and Q2 not in Q2s:
                    Q2s.add(Q2)
                    
            for Q2 in Q2s:
                s = self.qd.sym_k[Q2]
                self.s = s
                U_cc = self.qd.symmetry.op_scc[s]
                time_reversal = self.qd.time_reversal_k[Q2]
                self.sign = 1 - 2 * time_reversal
                Q_c = self.qd.bzk_kc[Q2]
                d_c = self.sign * np.dot(U_cc, q_c) - Q_c
                assert np.allclose(d_c.round(), d_c)
              
                for u1, kpt1 in enumerate(mykpts):
                    K2 = kd.find_k_plus_q(Q_c, [kpt1.K])[0] 
                    kpt2 = self.get_k_point(kpt1.s, K2, 0, self.nbands, 
                                            block=True)
                    k1 = kd.bz2ibz_k[kpt1.K]
                    i = self.kpts.index(k1)

                    N_c = pd0.gd.N_c
                    i_cG = self.sign * np.dot(U_cc, 
                                              np.unravel_index(pd0.Q_qG[0], 
                                                               N_c))

                    k1_c = kd.bzk_kc[kpt1.K]
                    k2_c = kd.bzk_kc[K2]
                    # This is the q that connects K1 and K2 in the 1st BZ
                    q1_c = k2_c - k1_c

                    # G-vector that connects the full Q_c with q1_c
                    shift1_c = q1_c - self.sign * np.dot(U_cc, q_c)
                    assert np.allclose(shift1_c.round(), shift1_c)
                    shift1_c = shift1_c.round().astype(int)
                    shift_c = kpt1.shift_c - kpt2.shift_c - shift1_c
                    I_G = np.ravel_multi_index(i_cG + shift_c[:, None], 
                                               N_c, 'wrap') 
                    pos_av = np.dot(self.spos_ac, pd0.gd.cell_cv)
                    M_vv = np.dot(pd0.gd.cell_cv.T,
                                  np.dot(U_cc.T,
                                         np.linalg.inv(pd0.gd.cell_cv).T))
                    Q_aGii = []
                    for a, Q_Gii in enumerate(self.Q_aGii):
                        x_G = np.exp(1j * np.dot(G_Gv, (pos_av[a] -
                                                        np.dot(M_vv, 
                                                               pos_av[a]))))
                        U_ii = self.calc.wfs.setups[a].R_sii[self.s]
                        Q_Gii = np.dot(np.dot(U_ii, Q_Gii * x_G[:, None, 
                                                                   None]),
                                       U_ii.T).transpose(1, 0, 2)
                        if self.sign == -1:
                            Q_Gii = Q_Gii.conj()
                        Q_aGii.append(Q_Gii)

                    for n in range(kpt1.n2 - kpt1.n1):
                        ut1cc_R = kpt1.ut_nR[n].conj()
                        eps1 = kpt1.eps_n[n]
                        C1_aGi = [np.dot(Qa_Gii, P1_ni[n].conj())
                                  for Qa_Gii, P1_ni in zip(Q_aGii, kpt1.P_ani)]

                        n_mG = self.calculate_pair_densities(ut1cc_R, C1_aGi, 
                                                             kpt2, pd0, I_G)
                        if self.sign == 1:
                            n_mG = n_mG.conj()
                                    
                        f_m = kpt2.f_n
                        deps_m = eps1 - kpt2.eps_n
                        sigma, dsigma = self.calculate_sigma(n_mG, deps_m, 
                                                             f_m, Wpm_w)
                        nn = kpt1.n1 + n - self.bands[0]
                        self.sigma_sin[kpt1.s, i, nn] += sigma
                        self.dsigma_sin[kpt1.s, i, nn] += dsigma
                       
        self.world.sum(self.sigma_sin)
        self.world.sum(self.dsigma_sin)

        self.complete = True
        self.save_state_file()
    
        return self.sigma_sin, self.dsigma_sin

    def calculate_qp_correction(self):

        if self.complete:
            print('Self-energy loaded from file', file=self.fd)
        else:
            self.calculate_QEH()  

        # Need GW result for renormalization factor
        b1, b2 = self.bands
        gwdata = pickle.load(open(self.gwfile, 'rb'))
        self.dsigmagw_sin = gwdata['dsigma']
        self.qpgw_sin = gwdata['qp'] / Hartree
        nk = self.qpgw_sin.shape[1]
        if not self.sigma_sin.shape[1] == nk:
            self.sigma_sin = np.repeat(self.sigma_sin[:, :1, :], nk, axis=1)
            self.dsigma_sin = np.repeat(self.dsigma_sin[:, :1, :], nk, axis=1)
        self.Z_sin = 1. / (1 - self.dsigma_sin - self.dsigmagw_sin)
        self.qp_sin = self.Z_sin * self.sigma_sin
        
        return self.qp_sin * Hartree
    
    def calculate_qp_energies(self):
        # calculate 
        qp_sin = self.calculate_qp_correction() / Hartree
        self.Qp_sin = self.qpgw_sin + qp_sin
        self.save_state_file()
        return self.Qp_sin * Hartree

    @timer('Sigma')
    def calculate_sigma(self, n_mG, deps_m, f_m, W_wGG):
        """Calculates a contribution to the self-energy and its derivative for
        a given (k, k-q)-pair from its corresponding pair-density and
        energy."""
        o_m = abs(deps_m)
        # Add small number to avoid zeros for degenerate states:
        sgn_m = np.sign(deps_m + 1e-15)
        
        # Pick +i*eta or -i*eta:
        s_m = (1 + sgn_m * np.sign(0.5 - f_m)).astype(int) // 2
        comm = self.blockcomm
        nw = len(self.omega_w)
        nG = n_mG.shape[1]
        mynG = (nG + comm.size - 1) // comm.size
        Ga = min(comm.rank * mynG, nG)
        Gb = min(Ga + mynG, nG)
        beta = (2**0.5 - 1) * self.domega0 / self.omega2
        w_m = (o_m / (self.domega0 + beta * o_m)).astype(int)
        o1_m = self.omega_w[w_m]
        o2_m = self.omega_w[w_m + 1]
        
        x = 1.0 / (self.qd.nbzkpts * 2 * pi * self.vol)
        sigma = 0.0
        dsigma = 0.0
        # Performing frequency integration
        for o, o1, o2, sgn, s, w, n_G in zip(o_m, o1_m, o2_m,
                                             sgn_m, s_m, w_m, n_mG):
        
            C1_GG = W_wGG[s*nw + w]
            C2_GG = W_wGG[s*nw + w + 1]
            p = x * sgn
            myn_G = n_G[Ga:Gb]
            sigma1 = p * np.dot(np.dot(myn_G, C1_GG), n_G.conj()).imag
            sigma2 = p * np.dot(np.dot(myn_G, C2_GG), n_G.conj()).imag
            sigma += ((o - o1) * sigma2 + (o2 - o) * sigma1) / (o2 - o1)
            dsigma += sgn * (sigma2 - sigma1) / (o2 - o1)

        return sigma, dsigma

    def save_state_file(self, q=0):
        data = {'kpts': self.kpts,
                'bands': self.bands,
                'nbands': self.nbands,
                'last_q': self.nq,
                'complete': self.complete,
                'sigma_sin': self.sigma_sin,
                'dsigma_sin': self.dsigma_sin,
                'qp_sin': self.qp_sin,
                'Qp_sin': self.Qp_sin}
        if self.world.rank == 0:
            np.savez(self.filename + '_qeh.npz',
                     **data)

    def load_state_file(self):
        try:
            data = np.load(self.filename + '_qeh.npz')
        except IOError:
            return False
        else:
            if (data['kpts'] == self.kpts and
                data['bands'] == self.bands and
                data['nbands'] == self.nbands):
                self.nq = data['last_q']
                self.complete = data['complete']
                self.complete = data['complete']
                self.sigma_sin = data['sigma_sin']
                self.dsigma_sin = data['dsigma_sin']
                return True
            else:
                return False    

    def get_W_on_grid(self, dW_qw, include_q0=True, metal=False):
        """This function transforms the screened potential W(q,w) to the 
        (q,w)-grid of the GW calculation. Also, W is integrated over
        a region around q=0 if include_q0 is set to True."""
    
        q_cs = self.qd.ibzk_kc

        rcell_cv = 2 * pi * np.linalg.inv(self.calc.wfs.gd.cell_cv).T
        q_vs = np.dot(q_cs, rcell_cv)
        q_grid = (q_vs**2).sum(axis=1)**0.5
        self.q_grid = q_grid
        w_grid = self.omega_w

        wqeh = self.wqeh  # w_grid.copy() # self.qeh
        qqeh = self.qqeh
        sortqeh = np.argsort(qqeh)
        qqeh = qqeh[sortqeh]
        dW_qw = dW_qw[sortqeh] 

        sort = np.argsort(q_grid)
        isort = np.argsort(sort)
        if metal and np.isclose(qqeh[0], 0):
            """We don't have the right q=0 limit for metals  and semi-metals.
            -> Point should be omitted from interpolation"""
            qqeh = qqeh[1:]
            dW_qw = dW_qw[1:]
            sort = sort[1:]
            
        from scipy.interpolate import RectBivariateSpline
        yr = RectBivariateSpline(qqeh, wqeh, dW_qw.real, s=0)
        yi = RectBivariateSpline(qqeh, wqeh, dW_qw.imag, s=0)

        dWgw_qw = yr(q_grid[sort], w_grid) + 1j * yi(q_grid[sort], w_grid)
        dW_qw = yr(qqeh, w_grid) + 1j * yi(qqeh, w_grid)
        
        if metal:
            # Interpolation is done -> put back zeros at q=0
            dWgw_qw = np.insert(dWgw_qw, 0, 0, axis=0)
            qqeh = np.insert(qqeh, 0, 0)
            dW_qw = np.insert(dW_qw, 0, 0, axis=0)
            q_cut = q_grid[sort][0] / 2.
        else:
            q_cut = q_grid[sort][1] / 2.
            
        q0 = np.array([q for q in qqeh if q <= q_cut])
        if len(q0) > 1:  # Integrate arround q=0 
            vol = np.pi * (q0[-1] + q0[1] / 2.)**2 
            if np.isclose(q0[0], 0):
                weight0 = np.pi * (q0[1] / 2.)**2 / vol
                c = (1 - weight0) / np.sum(q0)
                weights = c * q0        
                weights[0] = weight0
            else:
                c = 1 / np.sum(q0)
                weights = c * q0        

            dWgw_qw[0] = (np.repeat(weights[:, np.newaxis], len(w_grid), 
                                    axis=1) * dW_qw[:len(q0)]).sum(axis=0)
        
        if not include_q0:  # Omit q=0 contrinution completely.
            dWgw_qw[0] = 0.0
             
        dWgw_qw = dWgw_qw[isort]  # Put dW back on native grid.
        return dWgw_qw

    def calculate_W_QEH(self, structure, d, layer=0):
        from gpaw.response.qeh import Heterostructure, expand_layers

        structure = expand_layers(structure)
        self.w_grid = self.omega_w
        wmax = self.w_grid[-1]
        # qmax = (self.q_grid).max()

        # Single layer
        s = (np.insert(d, 0, d[0]) +
             np.append(d, d[-1])) / 2.
        d0 = s[layer]
        HS0 = Heterostructure(structure=[structure[layer]], 
                              d=[],
                              d0=d0,
                              wmax=wmax * Hartree,
                              # qmax=qmax / Bohr
                              ) 

        W0_qw = HS0.get_screened_potential()
        
        # Full heterostructure
      
        HS = Heterostructure(structure=structure, d=d,
                             wmax=wmax * Hartree,
                             # qmax=qmax / Bohr
                             ) 
        W_qw = HS.get_screened_potential(layer=layer)
    
        # Difference in screened potential:
        dW_qw = W_qw - W0_qw
        self.wqeh = HS.frequencies
        self.qqeh = HS.q_abs

        if self.world.rank == 0:
            data = {'qqeh': self.qqeh,
                    'wqeh': self.wqeh,
                    'dW_qw': dW_qw}
            np.savez(self.filename + "_dW_qw.npz",
                     **data)

        return dW_qw

    def find_maximum_frequency(self):
        self.epsmin = 10000.0
        self.epsmax = -10000.0
        for kpt in self.calc.wfs.kpt_u:
            self.epsmin = min(self.epsmin, kpt.eps_n[0])
            self.epsmax = max(self.epsmax, kpt.eps_n[self.nbands - 1])
            
        print('Minimum eigenvalue: %10.3f eV' % (self.epsmin * Hartree),
              file=self.fd)
        print('Maximum eigenvalue: %10.3f eV' % (self.epsmax * Hartree),
              file=self.fd)

        return self.epsmax - self.epsmin
