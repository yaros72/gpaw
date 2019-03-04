import numpy as np

import ase.units

from gpaw.lrtddft2.ks_singles import KohnShamSingleExcitation
from gpaw.lrtddft2.lr_layouts import LrTDDFPTSolveLayout


class LrResponse:
    def __init__(self, lrtddft2, excitation_energy, field_vector, lorentzian_width, sl_lrtddft = None):
        self.lrtddft2 = lrtddft2
        self.omega = excitation_energy
        self.eta = lorentzian_width
        self.field_vector = np.array(field_vector)
        self.sl_lrtddft = sl_lrtddft

        self.C_re = None
        self.C_im = None

        self.response_wf_ready = False

    def read(self):
        raise RuntimeError('Error in LrtddftResponseWF read: Not implemented')

    # TD-DFPT:
    #
    # A C = -Vext
    #
    # ( E+KN+w    KN     -eta     0    ) ( C_+w,Re )   ( -dm_laser )
    # (   KN    E+KN-w    0      -eta  ) ( C_-w,Re ) = ( -dm_laser )
    # (  eta      0     E+KN+w   -KN   ) ( C_+w,Im )   (      0    )
    # (   0      eta     -KN    E+KN-w ) ( C_-w,Im )   (      0    )
    #
    # N is occupation difference matrix (just like in Casida)
    #    (at least K = 0 gives density diffs for both real and imaginary part)
    def solve(self):
        if self.sl_lrtddft is None:
            self.solve_serial()
        else:
            self.solve_parallel()

    def get_response_coefficients(self, units = 'eVang'):
        C_re = self.C_re * 1.
        C_im = self.C_im * 1.

        if units == 'au':
            pass
        elif units == 'eVang':
            # FIXME: where these units come from???
            # S = 2 omega detla n C_im dm
            # units: 1/eV = eV * e ang * C
            # => units of C: 1/(eV * eV * e ang)
            # maybe
            # psi(w) = psi0(w) + E(w).mu psi1(w) + ...
            # E(w).mu units = V/ang e ang = eV ?
            # nope...
            C_re *= 1./(ase.units.Hartree**2 * ase.units.Bohr)
            C_im *= 1./(ase.units.Hartree**2 * ase.units.Bohr)
        else:
            raise RuntimeError('Error in get_response_coefficients: Invalid units.')

        return (C_re, C_im)

    def get_response_data(self, units = 'eVangcgs'):
        """Get response data.

        Returns matrix where transitions are in rows and columns are
        occupied index, unoccupied index, occupied KS eigenvalue, unoccupied
        KS eigenvalue, occupied occupation number, unoccupied occupation number,
        real part of response coefficent, imaginary part of response
        coefficient, dipole moment x, y, and z-components, magnetic moment x, y,
        and z-components.
        """

        data = np.zeros((len(self.lrtddft2.ks_singles.kss_list), 2+2+2+2+3+3+3))
        kpt_ind = self.lrtddft2.kpt_ind

        for (ip,kss_ip) in enumerate(self.lrtddft2.ks_singles.kss_list):
            i = kss_ip.occ_ind
            p = kss_ip.unocc_ind

            eps_i = self.lrtddft2.calc.wfs.kpt_u[kpt_ind].eps_n[i]
            eps_p = self.lrtddft2.calc.wfs.kpt_u[kpt_ind].eps_n[p]

            f_i = self.lrtddft2.calc.wfs.kpt_u[kpt_ind].f_n[i]
            f_p = self.lrtddft2.calc.wfs.kpt_u[kpt_ind].f_n[p]

            data[ip,0] = i
            data[ip,1] = p

            data[ip,2] = eps_i
            data[ip,3] = eps_p

            data[ip,4] = f_i
            data[ip,5] = f_p

            data[ip,6] = self.C_re[ip]
            data[ip,7] = self.C_im[ip]

            data[ip,8]  = kss_ip.dip_mom_r[0]
            data[ip,9]  = kss_ip.dip_mom_r[1]
            data[ip,10] = kss_ip.dip_mom_r[2]

            data[ip,11] = kss_ip.magn_mom[0]
            data[ip,12] = kss_ip.magn_mom[1]
            data[ip,13] = kss_ip.magn_mom[2]

            df = f_i - f_p

            C_im_ip = self.C_im[ip]
            data[ip,14] = df * C_im_ip**2

            dm = np.vdot(kss_ip.dip_mom_r, self.field_vector)
            data[ip,15] = 2. * self.omega * df * C_im_ip * dm
            #data[ip,15] = 2. * (eps_p-eps_i) * df * C_im_ip * dm

            ### FIXME: IS THIS CORRECT ???
            # maybe answer is here
            # Varsano et al., Phys.Chem.Chem.Phys. 11, 4481 (2009)
            mm = np.vdot(kss_ip.magn_mom, self.field_vector)
            data[ip,16] = - df * C_im_ip * mm


        if units == 'au':
            pass
        elif units == 'eVangcgs':
            data[:,2] *= ase.units.Hartree
            data[:,3] *= ase.units.Hartree
            data[:,6] *= 1./(ase.units.Hartree**2 * ase.units.Bohr)
            data[:,7] *= 1./(ase.units.Hartree**2 * ase.units.Bohr)
            data[:,8] *= ase.units.Bohr
            data[:,9] *= ase.units.Bohr
            data[:,10] *= ase.units.Bohr
            data[:,11] *= float('NaN') # FIXME: units?
            data[:,12] *= float('NaN') # FIXME: units?
            data[:,13] *= float('NaN') # FIXME: units?
            data[:,14] *= float('NaN') # FIXME: units?
            data[:,15] *= float('NaN') # FIXME: units?
            data[:,16] *= float('NaN') # FIXME: units?
            raise RuntimeError('Error in get_response_data: Unit conversion from atomic units to eV ang cgs not implemented yet.')
        else:
            raise RuntimeError('Error in get_response_data: Invalid units.')

        return data

    # amplitude, dipole, rotatory
    # dipole should integrate to absorption spectrum
    # rotatory should integrate to CD spectrum
    def get_transition_contribution_maps(self, occ_min_energy, occ_max_energy, unocc_min_energy, unocc_max_energy, width = 0.05, energy_step = 0.05, units='eVangcgs'):
        data = self.get_response_data(units='au')

        eps_i  = data[:,2]
        eps_p  = data[:,3]
        a_ip   = data[:,14]
        s_ip   = data[:,15]
        r_ip   = data[:,16]

        # occupied energy range
        wx = np.arange(occ_min_energy, occ_max_energy, energy_step)
        # unoccupied energy range
        wy = np.arange(unocc_min_energy, unocc_max_energy, energy_step)

        A = np.outer(wx,wy) * 0.
        S = np.outer(wx,wy) * 0.
        R = np.outer(wx,wy) * 0.

        # A(omega_x, omega_y) =
        #   sum_ip delta n_ip |C^(im)_ip|**2
        #            * g(omega_x - eps_i) g(omega_y - eps_p)
        #
        # D(omega_x, omega_y) =
        #   sum_ip 2 * delta n_ip C^(im)_ip * mu_ip
        #            * g(omega_x - eps_i) g(omega_y - eps_p)
        #
        # where g(x) is gaussian
        for (ei, ep, a, s, r) in zip(eps_i, eps_p, a_ip, s_ip, r_ip):
            gx = np.exp( (-.5/width/width) * np.power(wx-ei,2) ) / width / np.sqrt(2*np.pi)
            gy = np.exp( (-.5/width/width) * np.power(wy-ep,2) ) / width / np.sqrt(2*np.pi)

            A += a * np.outer(gx,gy)
            S += s * np.outer(gx,gy)
            R += r * np.outer(gx,gy)

        # FIXME: units of A, S, and R

        if units == 'au':
            pass
        elif units == 'eVangcgs':
            raise RuntimeError('Error in get_transition_contribution_maps: Unit conversion from atomic units to eV ang cgs not implemented yet.')
        else:
            raise RuntimeError('Error in get_transition_contribution_maps: Invalid units.')

        return (wx,wy,A,S,R)

    def get_induced_density(self, units='au', collect=False, amplitude_filter=1e-5):
        # Init pair densities
        dnt_Gip = self.lrtddft2.calc.wfs.gd.empty()
        dnt_gip = self.lrtddft2.calc.density.finegd.empty()
        drhot_gip = self.lrtddft2.calc.density.finegd.empty()

        drhot_g = self.lrtddft2.calc.density.finegd.empty()
        drhot_g[:] = 0.0

        C_im = self.C_im

        maxC = np.max(abs(C_im))
        #print >> self.txt, maxC

        for (ip,kss_ip) in enumerate(self.lrtddft2.ks_singles.kss_list):
            # distribute ips over eh comm
            if self.lrtddft2.lr_comms.get_local_eh_index(ip) is None:
                continue
            if abs(C_im[ip]) < amplitude_filter * maxC:
                continue
            dnt_Gip[:] = 0.0
            dnt_gip[:] = 0.0
            drhot_gip[:] = 0.0
            kss_ip.calculate_pair_density( dnt_Gip, dnt_gip, drhot_gip )
            #### ?FIXME?: SHOULD HERE BE OCCUPATION DIFFERENCE? yes ####
            drhot_g += kss_ip.pop_diff * C_im[ip] * drhot_gip

        # sum over eh comm
        self.lrtddft2.lr_comms.eh_comm.sum(drhot_g)

        if collect:
            drhot_g = self.lrtddft2.calc.density.finegd.collect(drhot_g)

        if units == 'au':
            pass
        elif units == 'ang':
            drhot_g *= 1./ ase.units.Bohr**3
        else:
            raise RuntimeError('Error in get_induced_density: Invalid units.')

        return drhot_g

    def get_approximate_electron_and_hole_densities(self, units='au', collect=False, amplitude_filter=1e-4):
        # Init pair densities
        dnt_Gip = self.lrtddft2.calc.wfs.gd.empty()
        dnt_gip = self.lrtddft2.calc.density.finegd.empty()
        drhot_gip = self.lrtddft2.calc.density.finegd.empty()

        drhot_ge = self.lrtddft2.calc.density.finegd.empty()
        drhot_ge[:] = 0.0
        drhot_gh = self.lrtddft2.calc.density.finegd.empty()
        drhot_gh[:] = 0.0


        #
        # Sum of Slater determinants:
        # diag ip ip     => |c_ip|**2 ( |psi_p|**2 - |psi_i]**2 )
        # offdiag ip iq  =>   c_ip c_iq psi_p psi_q
        # offdiag ip jp  => - c_ip c_jp psi_i psi_j
        # offdiag ip jq  => 0
        #
        # remember both ip,jq  and jq,ip
        #

        # occupations for electron and hole
        f_n  = self.lrtddft2.calc.wfs.kpt_u[self.lrtddft2.kpt_ind].f_n
        fe_n = np.zeros(len(f_n))
        fh_n = np.zeros(len(f_n))

        C_im = self.C_im

        maxC = np.max(abs(C_im))

        # diagonal ip,ip
        for (ip,kss_ip) in enumerate(self.lrtddft2.ks_singles.kss_list):
            if self.lrtddft2.lr_comms.get_local_eh_index(ip) is None:
                continue
            if abs(C_im[ip]) < amplitude_filter * maxC:
                continue

            # decrease in density
            fh_n[kss_ip.occ_ind]   -= C_im[ip] * C_im[ip] * kss_ip.pop_diff
            # increase in density
            fe_n[kss_ip.unocc_ind] += C_im[ip] * C_im[ip] * kss_ip.pop_diff


        for k in range(len(f_n)):
            if ( abs(fh_n[k]) < amplitude_filter * maxC*maxC and
                 abs(fe_n[k]) < amplitude_filter * maxC*maxC ):
                continue

            dnt_Gip[:] = 0.0
            dnt_gip[:] = 0.0
            drhot_gip[:] = 0.0

            kss_ip = KohnShamSingleExcitation( self.lrtddft2.calc,
                                               self.lrtddft2.kpt_ind,
                                               k, k )
            kss_ip.calculate_pair_density( dnt_Gip, dnt_gip, drhot_gip )

            # fh_n < 0
            drhot_gh += fh_n[k] * drhot_gip
            # fe_n > 0
            drhot_ge += fe_n[k] * drhot_gip

            #if self.parent_comm.rank == 0:
            #    print '%03d => %03d : %03d=>%03d | %12.6lf  %12.6lf %12.6lf' % (kss_ip.occ_ind, kss_ip.unocc_ind, kss_ip.occ_ind, kss_ip.unocc_ind, C_im[ip], C_im[ip], C_im[ip] * C_im[ip])
            #    sys.stdout.flush()


        # offdiagonal
        for (ip,kss_ip) in enumerate(self.lrtddft2.ks_singles.kss_list):
            if self.lrtddft2.lr_comms.get_local_eh_index(ip) is None:
                continue
            if abs(C_im[ip]) < amplitude_filter * maxC:
                continue

            for (jq,kss_jq) in enumerate(self.lrtddft2.ks_singles.kss_list):
                # if diagonal, skip because already done
                if ( kss_ip.occ_ind   == kss_jq.occ_ind and
                     kss_ip.unocc_ind == kss_jq.unocc_ind ):
                    continue

                # if both are different, then orthogonal
                if ( kss_ip.occ_ind   != kss_jq.occ_ind and
                     kss_ip.unocc_ind != kss_jq.unocc_ind ):
                    continue

                if abs(C_im[ip] * C_im[jq]) < amplitude_filter * maxC * maxC:
                    continue


                # if occupied state is the same, then only electron density
                # changes (ie use kss_??.unocc_ind)
                # this gives direction, and should integrate to zero
                if ( kss_ip.occ_ind   == kss_jq.occ_ind and
                     kss_ip.unocc_ind != kss_jq.unocc_ind ):
                    dnt_Gip[:] = 0.0
                    dnt_gip[:] = 0.0
                    drhot_gip[:] = 0.0

                    kss_pq = KohnShamSingleExcitation( self.lrtddft2.calc,
                                                       self.lrtddft2.kpt_ind,
                                                       kss_ip.unocc_ind,
                                                       kss_jq.unocc_ind )
                    kss_pq.calculate_pair_density( dnt_Gip, dnt_gip, drhot_gip )

                    drhot_ge += (C_im[ip] * C_im[jq] * np.sqrt(kss_ip.pop_diff * kss_jq.pop_diff)) * drhot_gip


                # if occupied state is the same, then only hole density
                # changes (ie use kss_??.occ_ind)
                # this gives direction, and should integrate to zero
                if ( kss_ip.occ_ind   != kss_jq.occ_ind and
                     kss_ip.unocc_ind == kss_jq.unocc_ind ):
                    dnt_Gip[:] = 0.0
                    dnt_gip[:] = 0.0
                    drhot_gip[:] = 0.0
                    kss_ij = KohnShamSingleExcitation( self.lrtddft2.calc,
                                                       self.lrtddft2.kpt_ind,
                                                       kss_ip.occ_ind,
                                                       kss_jq.occ_ind )
                    kss_ij.calculate_pair_density( dnt_Gip, dnt_gip, drhot_gip )

                    drhot_gh -= (C_im[ip] * C_im[jq] * np.sqrt(kss_ip.pop_diff * kss_jq.pop_diff)) * drhot_gip



        self.lrtddft2.lr_comms.eh_comm.sum(drhot_ge)
        self.lrtddft2.lr_comms.eh_comm.sum(drhot_gh)

        drhot_geh = drhot_ge + drhot_gh

        #Ige  = self.lrtddft2.calc.density.finegd.integrate(drhot_ge)
        #Igh  = self.lrtddft2.calc.density.finegd.integrate(drhot_gh)
        #Igeh = self.lrtddft2.calc.density.finegd.integrate(drhot_geh)

        #if self.lrtddft2.lr_comms.parent_comm.rank == 0:
        #    print 'drho_ge ', Ige
        #    print 'drho_gh ', Igh
        #    print 'drho_geh', Igeh


        if collect:
            drhot_ge = self.lrtddft2.calc.density.finegd.collect(drhot_ge)
            drhot_gh = self.lrtddft2.calc.density.finegd.collect(drhot_gh)
            drhot_geh = self.lrtddft2.calc.density.finegd.collect(drhot_geh)


        if units == 'au':
            pass
        elif units == 'ang':
            drhot_geh *= 1./ ase.units.Bohr**3
            drhot_gh  *= 1./ ase.units.Bohr**3
            drhot_ge  *= 1./ ase.units.Bohr**3
        else:
            raise RuntimeError('Error in get_approximate_electron_and_hole_densities: Invalid units.')

        return (drhot_ge, drhot_gh, drhot_geh)

    def solve_serial(self):
        nrows = len(self.lrtddft2.ks_singles.kss_list)

        A_matrix = np.zeros((nrows*4, nrows*4))
        K_matrix = self.lrtddft2.K_matrix.values

        # add K N
        # slow but if you're using serial it's ok anyway
        for (ip, kss_ip) in enumerate(self.lrtddft2.ks_singles.kss_list):
            for (jq, kss_jq) in enumerate(self.lrtddft2.ks_singles.kss_list):
                lip = self.lrtddft2.lr_comms.get_local_eh_index(ip)
                ljq = self.lrtddft2.lr_comms.get_local_dd_index(jq)
                if lip is None or ljq is None:
                    continue

                A_matrix[ip*4+0,jq*4+0] =  K_matrix[lip,ljq] * kss_jq.pop_diff
                A_matrix[ip*4+0,jq*4+1] =  K_matrix[lip,ljq] * kss_jq.pop_diff
                A_matrix[ip*4+1,jq*4+0] =  K_matrix[lip,ljq] * kss_jq.pop_diff
                A_matrix[ip*4+1,jq*4+1] =  K_matrix[lip,ljq] * kss_jq.pop_diff

                A_matrix[ip*4+2,jq*4+2] =  K_matrix[lip,ljq] * kss_jq.pop_diff
                A_matrix[ip*4+2,jq*4+3] = -K_matrix[lip,ljq] * kss_jq.pop_diff
                A_matrix[ip*4+3,jq*4+2] = -K_matrix[lip,ljq] * kss_jq.pop_diff
                A_matrix[ip*4+3,jq*4+3] =  K_matrix[lip,ljq] * kss_jq.pop_diff

        # diagonal stuff: E, w, and eta
        for (ip, kss_ip) in enumerate(self.lrtddft2.ks_singles.kss_list):
            lip = self.lrtddft2.lr_comms.get_local_eh_index(ip)
            ljq = self.lrtddft2.lr_comms.get_local_dd_index(ip)
            if lip is None or ljq is None:
                continue

            # energy difference and excitation energy
            A_matrix[ip*4+0,ip*4+0] += kss_ip.energy_diff + self.omega
            A_matrix[ip*4+1,ip*4+1] += kss_ip.energy_diff - self.omega
            A_matrix[ip*4+2,ip*4+2] += kss_ip.energy_diff + self.omega
            A_matrix[ip*4+3,ip*4+3] += kss_ip.energy_diff - self.omega

            # life-time parameter
            A_matrix[ip*4+0,ip*4+2] += -self.eta
            A_matrix[ip*4+1,ip*4+3] += -self.eta
            A_matrix[ip*4+2,ip*4+0] +=  self.eta
            A_matrix[ip*4+3,ip*4+1] +=  self.eta


        # RHS
        V_rhs = np.zeros(nrows*4)
        for (jq, kss_jq) in enumerate(self.lrtddft2.ks_singles.kss_list):
            ljq = self.lrtddft2.lr_comms.get_local_dd_index(jq)
            if ljq is None:
                continue

            # 2 cos(wt) = exp(+iwt) + exp(iwt)
            # fixme: no minus here???
            # -dm_laser = - (-er) cos(wt) ????
            V_rhs[jq*4+0] = np.dot(kss_jq.dip_mom_r, self.field_vector)
            V_rhs[jq*4+1] = np.dot(kss_jq.dip_mom_r, self.field_vector)
            V_rhs[jq*4+2] = 0.
            V_rhs[jq*4+3] = 0.


        # no transpose
        #A_matrix[:] = A_matrix.transpose()


        # solve A C = V
        self.lrtddft2.lr_comms.parent_comm.sum(A_matrix)
        self.lrtddft2.lr_comms.dd_comm.sum(V_rhs)

        #np.set_printoptions(precision=5, suppress=True)
        #print A_matrix
        C = np.linalg.solve(A_matrix,V_rhs)
        #print C

        # response wavefunction
        #   perturbation was exp(iwt) + exp(-iwt) = 2 cos(wt)
        #   => real part of the polarizability is propto 2 cos(wt)
        #   => imag part of the polarizability is propto 2 sin(wt)
        #
        # exp(iwt)
        # dn+ = phi-**h phi0 + phi0**h phi+ = phi0 (phi-**h + phi+)
        # exp(-iwt)
        # dn- = phi+**h phi0 + phi0**h phi- = phi0 (phi+**h + phi-) = dn+**h
        #
        # 2 cos(wt)
        # dn_2cos =    dn+ + dn-      =     dn+ + dn+**h  = Re[dn+] + Re[dn-]
        # 2 sin(wt)
        # dn_2sin = -i(dn+ - dn-)     = -i (dn+ - dn-**h) = Im[dn+] - Im[dn-]

        self.C_re = np.zeros(len(self.lrtddft2.ks_singles.kss_list))
        self.C_im = np.zeros(len(self.lrtddft2.ks_singles.kss_list))

        for (jq, kss_jq) in enumerate(self.lrtddft2.ks_singles.kss_list):
            # Re[phi-**h + phi_+]xo
            self.C_re[jq] = C[jq*4+0] + C[jq*4+1]
            # Im[phi-**h + phi_+]
            self.C_im[jq] = C[jq*4+2] - C[jq*4+3]

        # normalization... where it comes from? fourier (or lorentzian)...
        self.C_re *= 1./np.pi
        self.C_im *= 1./np.pi

        # anyway, you can check that
        #
        # osc_z(omega) / (pi eta) =
        #    2 * omega * sum_ip n_ip C^(im)_ip(omega) * mu^(z)_ip
        #
        # where osc_z, you can get from Casida
        # and 1/(pi eta) is because of Lorentzian folding

        # spectrum
        # S_z(omega) = 2 * omega sum_ip n_ip C^(im)_ip(omega) * mu^(z)_ip

        return (self.C_re,self.C_im)

    def solve_parallel(self):
        # total rows
        nrows = len(self.lrtddft2.ks_singles.kss_list)
        # local
        nlrows = self.lrtddft2.K_matrix.values.shape[0]
        nlcols = self.lrtddft2.K_matrix.values.shape[1]

        # create BLACS layout
        layout = LrTDDFPTSolveLayout( self.sl_lrtddft, nrows, self.lrtddft2.lr_comms )
        if ( nrows < np.max([layout.mprocs,layout.nprocs]) * layout.block_size ):
            raise RuntimeError('Linear response matrix is not large enough for the given number of processes (or block size) in sl_lrtddft. Please, use less processes (or smaller block size).')

        # build local part
        # NOTE: scalapack needs TRANSPOSED matrix!!!
        K_matrix = self.lrtddft2.K_matrix.values
        KN_matrix_T = np.zeros([nlcols, nlrows])
        A_matrix_T = np.zeros((nlcols*4, nlrows*4))

        KN_matrix_T[:] = K_matrix.T
        for (jq, kss_jq) in enumerate(self.lrtddft2.ks_singles.kss_list):
            ljq = self.lrtddft2.lr_comms.get_local_dd_index(jq)
            if ljq is None:
                continue
            KN_matrix_T[ljq,:] = K_matrix[:,ljq] * kss_jq.pop_diff


        lip_list = []
        ljq_list = []
        for (ip, kss_ip) in enumerate(self.lrtddft2.ks_singles.kss_list):
            lip = self.lrtddft2.lr_comms.get_local_eh_index(ip)
            ljq = self.lrtddft2.lr_comms.get_local_dd_index(ip)
            if lip is not None:
                lip_list.append(lip)
            if ljq is not None:
                ljq_list.append(ljq)

        lip4_list = np.array(lip_list)*4

        # add K N
        for ljq in ljq_list:
            A_matrix_T[ljq*4+0,lip4_list+0] =  KN_matrix_T[ljq,:]
            A_matrix_T[ljq*4+0,lip4_list+1] =  KN_matrix_T[ljq,:]
            A_matrix_T[ljq*4+1,lip4_list+0] =  KN_matrix_T[ljq,:]
            A_matrix_T[ljq*4+1,lip4_list+1] =  KN_matrix_T[ljq,:]

            A_matrix_T[ljq*4+2,lip4_list+2] =  KN_matrix_T[ljq,:]
            A_matrix_T[ljq*4+2,lip4_list+3] =  -KN_matrix_T[ljq,:]
            A_matrix_T[ljq*4+3,lip4_list+2] =  -KN_matrix_T[ljq,:]
            A_matrix_T[ljq*4+3,lip4_list+3] =  KN_matrix_T[ljq,:]


        # diagonal stuff: E, w, and eta
        for (ip, kss_ip) in enumerate(self.lrtddft2.ks_singles.kss_list):
            lip = self.lrtddft2.lr_comms.get_local_eh_index(ip)
            ljq = self.lrtddft2.lr_comms.get_local_dd_index(ip)
            if lip is None or ljq is None:
                continue

            # energy difference and excitation energy
            A_matrix_T[ljq*4+0,lip*4+0] += kss_ip.energy_diff + self.omega
            A_matrix_T[ljq*4+1,lip*4+1] += kss_ip.energy_diff - self.omega
            A_matrix_T[ljq*4+2,lip*4+2] += kss_ip.energy_diff + self.omega
            A_matrix_T[ljq*4+3,lip*4+3] += kss_ip.energy_diff - self.omega

            # life-time parameter, transposed again!
            A_matrix_T[ljq*4+0,lip*4+2] +=  self.eta
            A_matrix_T[ljq*4+1,lip*4+3] +=  self.eta
            A_matrix_T[ljq*4+2,lip*4+0] += -self.eta
            A_matrix_T[ljq*4+3,lip*4+1] += -self.eta


        # RHS
        V_rhs = []
        for (jq, kss_jq) in enumerate(self.lrtddft2.ks_singles.kss_list):
            if jq % self.lrtddft2.lr_comms.parent_comm.size != self.lrtddft2.lr_comms.parent_comm.rank:
                continue

            # 2 cos(wt) = exp(+iwt) + exp(iwt)
            # fixme: no minus here???
            # -dm_laser = - (-er) cos(wt) ????
            V_rhs.append( np.dot(kss_jq.dip_mom_r, self.field_vector) )
            V_rhs.append( np.dot(kss_jq.dip_mom_r, self.field_vector) )
            V_rhs.append( 0. )
            V_rhs.append( 0. )


        # debug
        #if False:
        #    A_matrix_T[:,:] = 0.
        #    for (ip, kss_ip) in enumerate(self.lrtddft2.ks_singles.kss_list):
        #        for (jq, kss_jq) in enumerate(self.lrtddft2.ks_singles.kss_list):
        #            lip = self.lrtddft2.lr_comms.get_local_eh_index(ip)
        #            ljq = self.lrtddft2.lr_comms.get_local_dd_index(jq)
        #            if lip is None or ljq is None:
        #                continue
        #
        #            if ip == jq:
        #                A_matrix_T[ljq*4+0,lip*4+0] += (ip+1)*100 + (jq+1)
        #                A_matrix_T[ljq*4+1,lip*4+1] += (ip+1)*100 + (jq+1)
        #                A_matrix_T[ljq*4+2,lip*4+2] += (ip+1)*100 + (jq+1)
        #                A_matrix_T[ljq*4+3,lip*4+3] += (ip+1)*100 + (jq+1)
        #            if ip == jq+1:
        #                A_matrix_T[ljq*4+0,lip*4+0] += (ip+1)*100 + (jq+1)
        #                A_matrix_T[ljq*4+1,lip*4+1] += (ip+1)*100 + (jq+1)
        #                A_matrix_T[ljq*4+2,lip*4+2] += (ip+1)*100 + (jq+1)
        #                A_matrix_T[ljq*4+3,lip*4+3] += (ip+1)*100 + (jq+1)
        #
        #    for i in range(self.lrtddft2.lr_comms.parent_comm.size):
        #        if ( self.lrtddft2.lr_comms.parent_comm.rank == i and
        #             len(A_matrix_T.ravel()) > 0 ):
        #            print('%d / %d : %d / %d  = %lf => %lf' % (self.lrtddft2.lr_comms.eh_comm.rank,
        #                                                       self.lrtddft2.lr_comms.eh_comm.size,
        #                                                       self.lrtddft2.lr_comms.dd_comm.rank,
        #                                                       self.lrtddft2.lr_comms.dd_comm.size,
        #                                                       A_matrix_T[0,0], A_matrix_T[-1,-1] ) )
        #        self.lrtddft2.lr_comms.parent_comm.barrier()
        #
        #    V_rhs = []
        #    for (jq, kss_jq) in enumerate(self.lrtddft2.ks_singles.kss_list):
        #        if jq % self.lrtddft2.lr_comms.parent_comm.size != self.lrtddft2.lr_comms.parent_comm.rank:
        #            continue
        #        V_rhs.append(1.*jq)
        #        V_rhs.append(1.*jq)
        #        V_rhs.append(1.*jq)
        #        V_rhs.append(1.*jq)

        # no transpose!
        #A_matrix[:] = A_matrix.transpose()
        V_rhs_T = np.array([V_rhs])

        # solve A C = V
        C = layout.solve(A_matrix_T, V_rhs_T)[0]

        # see solve serial for comments
        self.C_re = np.zeros(len(self.lrtddft2.ks_singles.kss_list))
        self.C_im = np.zeros(len(self.lrtddft2.ks_singles.kss_list))

        l = 0
        for (jq, kss_jq) in enumerate(self.lrtddft2.ks_singles.kss_list):
            if jq % self.lrtddft2.lr_comms.parent_comm.size != self.lrtddft2.lr_comms.parent_comm.rank:
                continue
            # Re[phi-**h + phi_+]
            self.C_re[jq] = C[l*4+0] + C[l*4+1]
            # Im[phi-**h + phi_+]
            self.C_im[jq] = C[l*4+2] - C[l*4+3]
            l += 1

        self.lrtddft2.lr_comms.parent_comm.sum(self.C_re)
        self.lrtddft2.lr_comms.parent_comm.sum(self.C_im)

        # normalization... where it comes from? fourier (or lorentzian)...
        self.C_re *= 1./np.pi
        self.C_im *= 1./np.pi

        return (self.C_re,self.C_im)


