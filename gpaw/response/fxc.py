# -*- coding: utf-8
"""This module calculates XC kernels for response function calculations.
"""
from __future__ import print_function

import numpy as np

from gpaw.xc import XC
from gpaw.sphere.lebedev import weight_n, R_nv
from gpaw.io.tar import Reader
from gpaw.wavefunctions.pw import PWDescriptor

from ase.utils.timing import Timer
from ase.units import Bohr, Ha


def get_xc_kernel(pd, chi0, functional='ALDA', kernel='density',
                  RSrep='gpaw',
                  chi0_wGG=None,
                  fxc_scaling=None,
                  density_cut=None,
                  spinpol_cut=None):
    """
    Factory function that calls the relevant functions below
    """

    if kernel == 'density':
        return get_density_xc_kernel(pd, chi0, functional=functional,
                                     RSrep=RSrep,
                                     chi0_wGG=chi0_wGG,
                                     density_cut=density_cut)
    elif kernel in ['+-', '-+']:
        # Currently only collinear adiabatic xc kernels are implemented
        # for which the +- and -+ kernels are the same
        return get_transverse_xc_kernel(pd, chi0, functional=functional,
                                        RSrep=RSrep,
                                        chi0_wGG=chi0_wGG,
                                        fxc_scaling=fxc_scaling,
                                        density_cut=density_cut,
                                        spinpol_cut=spinpol_cut)
    else:
        raise ValueError('%s kernels not implemented' % kernel)


def get_density_xc_kernel(pd, chi0, functional='ALDA',
                          RSrep='gpaw',
                          chi0_wGG=None,
                          density_cut=None):
    """
    Density-density xc kernels
    Factory function that calls the relevant functions below
    """

    calc = chi0.calc
    fd = chi0.fd
    nspins = len(calc.density.nt_sG)
    assert nspins == 1

    if functional[0] == 'A':
        # Standard adiabatic kernel
        print('Calculating %s kernel' % functional, file=fd)
        Kcalc = AdiabaticDensityKernelCalculator(fd, chi0.world, RSrep,
                                                 ecut=chi0.ecut,
                                                 density_cut=density_cut)
        Kxc_sGG = np.array([Kcalc(pd, calc, functional)])
    elif functional[0] == 'r':
        # Renormalized kernel
        print('Calculating %s kernel' % functional, file=fd)
        Kxc_sGG = calculate_renormalized_kernel(pd, calc, functional, fd)
    elif functional[:2] == 'LR':
        print('Calculating LR kernel with alpha = %s' % functional[2:],
              file=fd)
        Kxc_sGG = calculate_lr_kernel(pd, calc, alpha=float(functional[2:]))
    elif functional == 'DM':
        print('Calculating DM kernel', file=fd)
        Kxc_sGG = calculate_dm_kernel(pd, calc)
    elif functional == 'Bootstrap':
        print('Calculating Bootstrap kernel', file=fd)
        Kxc_sGG = get_bootstrap_kernel(pd, chi0, chi0_wGG, fd)
    else:
        raise ValueError('density-density %s kernel not'
                         + ' implemented' % functional)

    return Kxc_sGG[0]


def get_transverse_xc_kernel(pd, chi0, functional='ALDA_x',
                             RSrep='gpaw',
                             chi0_wGG=None,
                             fxc_scaling=None,
                             density_cut=None,
                             spinpol_cut=None):
    """ +-/-+ xc kernels
    Currently only collinear ALDA kernels are implemented
    Factory function that calls the relevant functions below
    """
    
    calc = chi0.calc
    fd = chi0.fd
    nspins = len(calc.density.nt_sG)
    assert nspins == 2
    
    if functional in ['ALDA_x', 'ALDA_X', 'ALDA']:
        # Adiabatic kernel
        print("Calculating transverse %s kernel" % functional, file=fd)
        Kcalc = AdiabaticTransverseKernelCalculator(fd, chi0.world, RSrep,
                                                    ecut=chi0.ecut,
                                                    density_cut=density_cut,
                                                    spinpol_cut=spinpol_cut)
    else:
        raise ValueError("%s spin kernel not implemented" % functional)
    
    Kxc_GG = Kcalc(pd, calc, functional)
    
    if fxc_scaling is not None:
        assert isinstance(fxc_scaling[0], bool)
        if fxc_scaling[0]:
            if fxc_scaling[1] is None:
                fxc_scaling[1] = find_Goldstone_scaling(pd, chi0,
                                                        chi0_wGG, Kxc_GG)
            
            assert isinstance(fxc_scaling[1], float)
            Kxc_GG *= fxc_scaling[1]
    
    return Kxc_GG


def find_Goldstone_scaling(pd, chi0, chi0_wGG, Kxc_GG):
    """ Find a scaling of the kernel to move the magnon peak to omeaga=0. """
    # q should be gamma - scale to hit Goldstone
    assert pd.kd.gamma
    
    fd = chi0.fd
    omega_w = chi0.omega_w
    wgs = np.abs(omega_w).argmin()

    if not np.allclose(omega_w[wgs], 0., rtol=1.e-8):
        raise ValueError("Frequency grid needs to include"
                         + " omega=0. to allow Goldstone scaling")

    fxcs = 1.
    print("Finding rescaling of kernel to fulfill the Goldstone theorem",
          file=fd)

    world = chi0.world
    # Only one rank, rgs, has omega=0 and finds rescaling
    nw = len(omega_w)
    mynw = (nw + world.size - 1) // world.size
    rgs, mywgs = wgs // mynw, wgs % mynw
    fxcsbuf = np.empty(1, dtype=float)
    if world.rank == rgs:
        chi0_GG = chi0_wGG[mywgs]
        chi_GG = np.dot(np.linalg.inv(np.eye(len(chi0_GG)) -
                                      np.dot(chi0_GG, Kxc_GG * fxcs)),
                        chi0_GG)
        # Scale so that kappaM=0 in the static limit (omega=0)
        kappaM = (chi0_GG[0, 0] / chi_GG[0, 0]).real
        # If kappaM > 0, increase scaling (recall: kappaM ~ 1 - Kxc Re{chi_0})
        scaling_incr = 0.1 * np.sign(kappaM)
        while abs(kappaM) > 1.e-7 and abs(scaling_incr) > 1.e-7:
            fxcs += scaling_incr
            if fxcs <= 0.0 or fxcs >= 10.:
                raise Exception('Found an invalid fxc_scaling of %.4f' % fxcs)

            chi_GG = np.dot(np.linalg.inv(np.eye(len(chi0_GG)) -
                                          np.dot(chi0_GG, Kxc_GG * fxcs)),
                            chi0_GG)
            kappaM = (chi0_GG[0, 0] / chi_GG[0, 0]).real

            # If kappaM changes sign, change sign and refine increment
            if np.sign(kappaM) != np.sign(scaling_incr):
                scaling_incr *= -0.2
        fxcsbuf[:] = fxcs

    # Broadcast found rescaling
    world.broadcast(fxcsbuf, rgs)
    fxcs = fxcsbuf[0]
    
    return fxcs


class AdiabaticKernelCalculator:
    """ Adiabatic kernels with PAW """
    
    def __init__(self, fd, world, RSrep, ecut=None):

        """
        RSrep : str
            real space representation of kernel ('gpaw' or 'grid')
        """
        
        self.fd = fd
        self.world = world
        self.RSrep = RSrep
        self.ecut = ecut
        
        self.permitted_functionals = []
        self.functional = None
    
    def __call__(self, pd, calc, functional):
        assert functional in self.permitted_functionals
        self.functional = functional
        add_fxc = self.add_fxc  # class methods not within the scope of call
        
        vol = pd.gd.volume
        npw = pd.ngmax
        
        if self.RSrep == 'grid':
            print("\tFinding all-electron density", file=self.fd)
            n_sG, gd = calc.density.get_all_electron_density(atoms=calc.atoms,
                                                             gridrefinement=1)
            qd = pd.kd
            lpd = PWDescriptor(self.ecut, gd, complex, qd,
                               gammacentered=pd.gammacentered)
            
            print("\tCalculating fxc on real space grid using"
                  + " all-electron density", file=self.fd)
            fxc_G = np.zeros(np.shape(n_sG[0]))
            add_fxc(gd, n_sG, fxc_G)
        else:
            nt_sG = calc.density.nt_sG
            gd, lpd = pd.gd, pd
            
            print("\tCalculating fxc on real space grid using smooth density",
                  file=self.fd)
            fxc_G = np.zeros(np.shape(nt_sG[0]))
            add_fxc(gd, nt_sG, fxc_G)
        
        print("\tFourier transforming into reciprocal space", file=self.fd)
        nG = gd.N_c
        nG0 = nG[0] * nG[1] * nG[2]
        
        tmp_g = np.fft.fftn(fxc_G) * vol / nG0
        
        Kxc_GG = np.zeros((npw, npw), dtype=complex)
        for iG, iQ in enumerate(lpd.Q_qG[0]):
            iQ_c = (np.unravel_index(iQ, nG) + nG // 2) % nG - nG // 2
            for jG, jQ in enumerate(lpd.Q_qG[0]):
                jQ_c = (np.unravel_index(jQ, nG) + nG // 2) % nG - nG // 2
                ijQ_c = (iQ_c - jQ_c)
                if (abs(ijQ_c) < nG // 2).all():
                    Kxc_GG[iG, jG] = tmp_g[tuple(ijQ_c)]
        
        if self.RSrep == 'gpaw':
            print("\tCalculating PAW corrections to the kernel", file=self.fd)
            
            G_Gv = pd.get_reciprocal_vectors()
            R_av = calc.atoms.positions / Bohr
            setups = calc.wfs.setups
            D_asp = calc.density.D_asp
            
            KxcPAW_GG = np.zeros_like(Kxc_GG)
            dG_GGv = np.zeros((npw, npw, 3))
            for v in range(3):
                dG_GGv[:, :, v] = np.subtract.outer(G_Gv[:, v], G_Gv[:, v])

            # Distribute computation of PAW correction equally among processes
            p_r = self.distribute_correction(setups, self.world.size)
            apdone = 0
            npdone = 0
            pdone = 0
            pdonebefore = np.sum(p_r[:self.world.rank])
            pdonenow = pdonebefore + p_r[self.world.rank]
            
            for a, setup in enumerate(setups):
                # PAW correction is evaluated on a radial grid
                Y_nL = setup.xc_correction.Y_nL
                rgd = setup.xc_correction.rgd

                # Continue if computation has been done already
                nn = len(Y_nL)
                ng = len(rgd.r_g)
                apdone += nn * ng
                if pdonebefore >= apdone or pdone >= pdonenow:
                    npdone += nn * ng
                    pdone += nn * ng
                    continue
                
                n_qg = setup.xc_correction.n_qg
                nt_qg = setup.xc_correction.nt_qg
                nc_g = setup.xc_correction.nc_g
                nct_g = setup.xc_correction.nct_g
                dv_g = rgd.dv_g

                D_sp = D_asp[a]
                B_pqL = setup.xc_correction.B_pqL
                D_sLq = np.inner(D_sp, B_pqL.T)
                nspins = len(D_sp)

                f_g = rgd.zeros()
                ft_g = rgd.zeros()

                n_sLg = np.dot(D_sLq, n_qg)
                nt_sLg = np.dot(D_sLq, nt_qg)

                # Add core density
                n_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nc_g
                nt_sLg[:, 0] += np.sqrt(4. * np.pi) / nspins * nct_g

                coefatoms_GG = np.exp(-1j * np.inner(dG_GGv, R_av[a]))
                
                for n, Y_L in enumerate(Y_nL):
                    # Continue if computation has been done already
                    npdone += ng
                    if pdonebefore >= npdone or pdone >= pdonenow:
                        pdone += ng
                        continue
                    
                    w = weight_n[n]

                    f_g[:] = 0.
                    n_sg = np.dot(Y_L, n_sLg)
                    add_fxc(rgd, n_sg, f_g)

                    ft_g[:] = 0.
                    nt_sg = np.dot(Y_L, nt_sLg)
                    add_fxc(rgd, nt_sg, ft_g)

                    dG_GG = np.inner(dG_GGv, R_nv[n])
                    for i in range(len(rgd.r_g)):
                        # Continue if previous ranks already did computation
                        pdone += 1
                        if pdonebefore >= pdone:
                            continue
                        # Do computation if needed
                        if pdone <= pdonenow:
                            coef_GG = np.exp(-1j * dG_GG * rgd.r_g[i])
                            KxcPAW_GG += w * coefatoms_GG\
                                * np.dot(coef_GG, (f_g[i] - ft_g[i])
                                         * dv_g[i])
            
            self.world.sum(KxcPAW_GG)
            Kxc_GG += KxcPAW_GG
        
        return Kxc_GG / vol

    def distribute_correction(self, setups, size):
        """ Make every process work an equal amount """
        # Figure out the total number of grid points
        tp = 0
        for a, setup in enumerate(setups):
            Y_nL = setup.xc_correction.Y_nL
            r_g = setup.xc_correction.rgd.r_g
            tp += len(Y_nL) * len(r_g)
        
        # How many points should each process compute
        ppr = tp // size
        p_r = []
        pdone = 0
        for rr in range(size):
            if pdone + ppr * (size - rr) > tp:
                ppr -= 1
            elif pdone + ppr * (size - rr) < tp:
                ppr += 1
            p_r.append(ppr)
            pdone += ppr

        return p_r
    
    def add_fxc(self, gd, n_sg, fxc_g):
        raise NotImplementedError


class AdiabaticDensityKernelCalculator(AdiabaticKernelCalculator):

    def __init__(self, fd, world, RSrep,
                 ecut=None,
                 density_cut=None):
        """
        density_cut : float
            cutoff density below which f_xc is set to zero
        """

        self.density_cut = density_cut
        
        AdiabaticKernelCalculator.__init__(self, fd, world, RSrep, ecut)

        self.permitted_functionals += ['ALDA_x', 'ALDA_X', 'ALDA']

    def __call__(self, pd, calc, functional):

        Kxc_GG = AdiabaticKernelCalculator.__call__(self, pd, calc,
                                                    functional)

        if pd.kd.gamma:
            Kxc_GG[0, :] = 0.0
            Kxc_GG[:, 0] = 0.0

        return Kxc_GG
        
    def add_fxc(self, gd, n_sG, fxc_G):
        """
        Calculate fxc, using the cutoffs from input above
        
        ALDA_x is an explicit algebraic version
        ALDA_X uses the libxc package
        """
        
        _calculate_fxc = self._calculate_fxc
        density_cut = self.density_cut
        
        # Mask small n
        n_G = np.sum(n_sG, axis=0)
        if density_cut:
            npos_G = np.abs(n_G) > density_cut
        else:
            npos_G = np.full(np.shape(n_G), True, np.array(True).dtype)

        # Calculate fxc
        fxc_G[npos_G] += _calculate_fxc(gd, n_sG)[npos_G]

    def _calculate_fxc(self, gd, n_sG):
        if self.functional == 'ALDA_x':
            n_G = np.sum(n_sG, axis=0)
            fx_G = -1. / 3. * (3. / np.pi)**(1. / 3.) * n_G**(-2. / 3.)
            return fx_G
        else:
            fxc_sG = np.zeros_like(n_sG)
            xc = XC(self.functional[1:])
            xc.calculate_fxc(gd, n_sG, fxc_sG)
                
            return fxc_sG[0]

        
class AdiabaticTransverseKernelCalculator(AdiabaticKernelCalculator):
    
    def __init__(self, fd, world, RSrep,
                 ecut=None,
                 density_cut=None,
                 spinpol_cut=None):
        """
        density_cut : float
            cutoff density below which f_xc is set to zero
        spinpol_cut : float
            cutoff spin polarization. Below, f_xc is evaluated in zeta=0 limit
        """
        
        self.density_cut = density_cut
        self.spinpol_cut = spinpol_cut
        
        AdiabaticKernelCalculator.__init__(self, fd, world, RSrep, ecut)

        self.permitted_functionals += ['ALDA_x', 'ALDA_X', 'ALDA']
    
    def add_fxc(self, gd, n_sG, fxc_G):
        """
        Calculate fxc, using the cutoffs from input above
        
        ALDA_x is an explicit algebraic version
        ALDA_X uses the libxc package
        """
        
        _calculate_pol_fxc = self._calculate_pol_fxc
        _calculate_unpol_fxc = self._calculate_unpol_fxc
        spinpol_cut = self.spinpol_cut
        density_cut = self.density_cut
        
        # Mask small zeta
        n_G, m_G = None, None
        if spinpol_cut is not None:
            m_G = n_sG[0] - n_sG[1]
            n_G = n_sG[0] + n_sG[1]
            zetasmall_G = np.abs(m_G / n_G) < spinpol_cut
        else:
            zetasmall_G = np.full(np.shape(n_sG[0]), False,
                                  np.array(False).dtype)
            
        # Mask small n
        if density_cut:
            if n_G is None:
                n_G = n_sG[0] + n_sG[1]
            npos_G = np.abs(n_G) > density_cut
        else:
            npos_G = np.full(np.shape(n_sG[0]), True, np.array(True).dtype)
        
        # Don't use small zeta limit if n is small
        zetasmall_G = np.logical_and(zetasmall_G, npos_G)
    
        # In small zeta limit, use unpolarized fxc
        if zetasmall_G.any():
            if n_G is None:
                n_G = n_sG[0] + n_sG[1]
            fxc_G[zetasmall_G] += _calculate_unpol_fxc(gd, n_G)[zetasmall_G]
        
        # Set fxc to zero if n is small
        allfine_G = np.logical_and(np.invert(zetasmall_G), npos_G)
        
        # Above both spinpol_cut and density_cut calculate polarized fxc
        if m_G is None:
            m_G = n_sG[0] - n_sG[1]
        fxc_G[allfine_G] += _calculate_pol_fxc(gd, n_sG, m_G)[allfine_G]
        
    def _calculate_pol_fxc(self, gd, n_sG, m_G):
        """ Calculate polarized fxc """
        
        assert np.shape(m_G) == np.shape(n_sG[0])
        
        if self.functional == 'ALDA_x':
            fx_G = - (6. / np.pi)**(1. / 3.) \
                * (n_sG[0]**(1. / 3.) - n_sG[1]**(1. / 3.)) / m_G
            return fx_G
        else:
            v_sG = np.zeros(np.shape(n_sG))
            xc = XC(self.functional[1:])
            xc.calculate(gd, n_sG, v_sg=v_sG)
                
            return (v_sG[0] - v_sG[1]) / m_G
    
    def _calculate_unpol_fxc(self, gd, n_G):
        """ Calculate unpolarized fxc """
        fx_G = - (3. / np.pi)**(1. / 3.) * 2. / 3. * n_G**(-2. / 3.)
        if self.functional in ('ALDA_x', 'ALDA_X'):
            return fx_G
        else:
            # From Perdew & Wang 1992
            A = 0.016887
            a1 = 0.11125
            b1 = 10.357
            b2 = 3.6231
            b3 = 0.88026
            b4 = 0.49671
            
            rs_G = 3. / (4. * np.pi) * n_G**(-1. / 3.)
            X_G = 2. * A * (b1 * rs_G**(1. / 2.)
                            + b2 * rs_G + b3 * rs_G**(3. / 2.) + b4 * rs_G**2.)
            ac_G = 2. * A * (1 + a1 * rs_G) * np.log(1. + 1. / X_G)
            
            fc_G = 2. * ac_G / n_G
            
            return fx_G + fc_G


def calculate_renormalized_kernel(pd, calc, functional, fd):
    """Renormalized kernel"""
    
    from gpaw.xc.fxc import KernelDens
    kernel = KernelDens(calc,
                        functional,
                        [pd.kd.bzk_kc[0]],
                        fd,
                        calc.wfs.kd.N_c,
                        None,
                        ecut=pd.ecut * Ha,
                        tag='',
                        timer=Timer())
    
    kernel.calculate_fhxc()
    r = Reader('fhxc_%s_%s_%s_%s.gpw' %
               ('', functional, pd.ecut * Ha, 0))
    Kxc_sGG = np.array([r.get('fhxc_sGsG')])
    
    v_G = 4 * np.pi / pd.G2_qG[0]
    Kxc_sGG[0] -= np.diagflat(v_G)
    
    if pd.kd.gamma:
        Kxc_sGG[:, 0, :] = 0.0
        Kxc_sGG[:, :, 0] = 0.0

    return Kxc_sGG


def calculate_lr_kernel(pd, calc, alpha=0.2):
    """Long range kernel: fxc = \alpha / |q+G|^2"""

    assert pd.kd.gamma

    f_G = np.zeros(len(pd.G2_qG[0]))
    f_G[0] = -alpha
    f_G[1:] = -alpha / pd.G2_qG[0][1:]

    return np.array([np.diag(f_G)])


def calculate_dm_kernel(pd, calc):
    """Density matrix kernel"""

    assert pd.kd.gamma

    nv = calc.wfs.setups.nvalence
    psit_nG = np.array([calc.wfs.kpt_u[0].psit_nG[n]
                        for n in range(4 * nv)])
    vol = np.linalg.det(calc.wfs.gd.cell_cv)
    Ng = np.prod(calc.wfs.gd.N_c)
    rho_GG = np.dot(psit_nG.conj().T, psit_nG) * vol / Ng**2

    maxG2 = np.max(pd.G2_qG[0])
    cut_G = np.arange(calc.wfs.pd.ngmax)[calc.wfs.pd.G2_qG[0] <= maxG2]

    G_G = pd.G2_qG[0]**0.5
    G_G[0] = 1.0

    Kxc_GG = np.diagflat(4 * np.pi / G_G**2)
    Kxc_GG = np.dot(Kxc_GG, rho_GG.take(cut_G, 0).take(cut_G, 1))
    Kxc_GG -= 4 * np.pi * np.diagflat(1.0 / G_G**2)

    return np.array([Kxc_GG])


def get_bootstrap_kernel(pd, chi0, chi0_wGG, fd):
    """ Bootstrap kernel (see below) """
    
    if chi0.world.rank == 0:
        chi0_GG = chi0_wGG[0]
        if chi0.world.size > 1:
            # If size == 1, chi0_GG is not contiguous, and broadcast()
            # will fail in debug mode.  So we skip it until someone
            # takes a closer look.
            chi0.world.broadcast(chi0_GG, 0)
    else:
        nG = pd.ngmax
        chi0_GG = np.zeros((nG, nG), complex)
        chi0.world.broadcast(chi0_GG, 0)

    return calculate_bootstrap_kernel(pd, chi0_GG, fd)


def calculate_bootstrap_kernel(pd, chi0_GG, fd):
    """Bootstrap kernel PRL 107, 186401"""

    if pd.kd.gamma:
        v_G = np.zeros(len(pd.G2_qG[0]))
        v_G[0] = 4 * np.pi
        v_G[1:] = 4 * np.pi / pd.G2_qG[0][1:]
    else:
        v_G = 4 * np.pi / pd.G2_qG[0]

    nG = len(v_G)
    K_GG = np.diag(v_G)

    fxc_GG = np.zeros((nG, nG), dtype=complex)
    dminv_GG = np.zeros((nG, nG), dtype=complex)

    for iscf in range(120):
        dminvold_GG = dminv_GG.copy()
        Kxc_GG = K_GG + fxc_GG

        chi_GG = np.dot(np.linalg.inv(np.eye(nG, nG)
                                      - np.dot(chi0_GG, Kxc_GG)), chi0_GG)
        dminv_GG = np.eye(nG, nG) + np.dot(K_GG, chi_GG)

        alpha = dminv_GG[0, 0] / (K_GG[0, 0] * chi0_GG[0, 0])
        fxc_GG = alpha * K_GG
        print(iscf, 'alpha =', alpha, file=fd)
        error = np.abs(dminvold_GG - dminv_GG).sum()
        if np.sum(error) < 0.1:
            print('Self consistent fxc finished in %d iterations !' % iscf,
                  file=fd)
            break
        if iscf > 100:
            print('Too many fxc scf steps !', file=fd)

    return np.array([fxc_GG])
