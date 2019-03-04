import numpy as np

import ase.units

import gpaw.utilities.lapack

from gpaw.lrtddft2.lr_layouts import LrDiagonalizeLayout


class LrtddftTransitions:
    def __init__(self, ks_singles, K_matrix, sl_lrtddft = None):
        self.ks_singles = ks_singles
        self.K_matrix = K_matrix
        self.lr_comms = self.ks_singles.lr_comms
        self.sl_lrtddft = sl_lrtddft
        self.eigenvalues = None
        self.eigenvectors = None
        self.trans_prop_ready = False

        # only for pros
        self.custom_axes = None

    ############################################################################
    def initialize(self):
        self.trans_prop_ready = False

    ############################################################################
    def read(self):
        pass

    ############################################################################
    def calculate(self):
        self.diagonalize()
        self.calculate_properties()

    ############################################################################
    def diagonalize(self):
        if self.sl_lrtddft is None:
            self.diagonalize_serial()
        else:
            self.diagonalize_scalapack()

        # now self.eigenvectors[iploc,kloc]
        # is the "iploc"th local element of
        # the "kloc"th local eigenvector

    def diagonalize_serial(self):
        nrows = len(self.ks_singles.kss_list)

        # build local part of the full omega matrix
        omega_matrix = np.zeros((nrows,nrows), dtype=float)
        for (ip, kss_ip) in enumerate(self.ks_singles.kss_list):
            for (jq, kss_jq) in enumerate(self.ks_singles.kss_list):
                lip = self.lr_comms.get_local_eh_index(ip)
                ljq = self.lr_comms.get_local_dd_index(jq)
                if lip is None or ljq is None:
                    continue

                # K-matrix
                #if self.K_matrix.file_format == 1:
                omega_matrix[ip,jq] = 2.* np.sqrt(kss_ip.energy_diff * kss_jq.energy_diff * kss_ip.pop_diff * kss_jq.pop_diff) * self.K_matrix.values[lip,ljq]
                # old Casida format ... doing this already in k_matrix.py
                #elif self.K_matrix.file_format == 0:
                #    omega_matrix[ip,jq] = self.K_matrix.values[lip,ljq]
                # invalid
                #else:
                #    raise RuntimeError('Invalid K-matrix file format')

                if ( ip == jq ):
                    omega_matrix[ip,jq] += kss_ip.energy_diff * kss_jq.energy_diff

        # full omega matrix to each process
        self.lr_comms.parent_comm.sum(omega_matrix)

        #if self.lr_comms.parent_comm.rank == 0:
        #    print omega_matrix

        # solve eigensystem
        self.eigenvalues = np.zeros(nrows)

        # debug
        #omega_matrix[:] = 0
        #for i in range(omega_matrix.shape[0]):
        #    for j in range(omega_matrix.shape[1]):
        #        if i == j: omega_matrix[i,j] = -2.
        #        if i+1 == j: omega_matrix[i,j] = omega_matrix[j,i] = 1.


        gpaw.utilities.lapack.diagonalize(omega_matrix, self.eigenvalues)

        # make columns have eigenvectors (not rows)
        omega_matrix = np.transpose(omega_matrix)

        # convert eigenvector back to local version
        nlrows = self.K_matrix.values.shape[0]
        nlcols = self.K_matrix.values.shape[1]

        # debug
        # (eh=2,dd=3)      (0,0)  (0,1)  (0,2)  (1,0)  (1,1)  (1,2)
        # 00 01 02 03      00 03    01     02   10 13    11     12
        # 10 11 12 13   => 20 23    21     22   30 33    31     32
        # 20 21 22 23
        # 30 31 32 33

        self.eigenvectors = np.zeros((nlrows,nlcols), dtype=float)
        for (ip, kss) in enumerate(self.ks_singles.kss_list):
            for (jq, kss) in enumerate(self.ks_singles.kss_list):
                lip = self.lr_comms.get_local_eh_index(ip)
                ljq = self.lr_comms.get_local_dd_index(jq)
                if lip is None or ljq is None:
                    continue
                self.eigenvectors[lip,ljq] = omega_matrix[ip,jq] # debug: ip*10 + jq

        #debug
        #if 0:
        #    for p in range(self.lr_comms.parent_comm.size):
        #        self.lr_comms.parent_comm.barrier()
        #        if p == self.lr_comms.parent_comm.rank:
        #            print self.lr_comms.parent_comm.rank, self.lr_comms.eh_comm.rank, self.lr_comms.dd_comm.rank
        #            for i in range(self.eigenvectors.shape[0]):
        #                for j in range(self.eigenvectors.shape[1]):
        #                    print '%02d' % int(self.eigenvectors[i,j]),
        #                print

    def diagonalize_scalapack(self):
        # total rows
        nrows = len(self.ks_singles.kss_list)
        # local
        nlrows = self.K_matrix.values.shape[0]
        nlcols = self.K_matrix.values.shape[1]

        # create BLACS layout
        layout = LrDiagonalizeLayout( self.sl_lrtddft, nrows, self.lr_comms )
        if ( nrows < np.max([layout.mprocs,layout.nprocs]) * layout.block_size ):
            raise RuntimeError('Linear response matrix is not large enough for the given number of processes (or block size) in sl_lrtddft. Please, use less processes (or smaller block size).')

        # build local part
        omega_matrix = np.zeros((nlrows,nlcols), dtype=float)
        for (ip, kss_ip) in enumerate(self.ks_singles.kss_list):
            for (jq, kss_jq) in enumerate(self.ks_singles.kss_list):
                lip = self.lr_comms.get_local_eh_index(ip)
                ljq = self.lr_comms.get_local_dd_index(jq)
                if lip is None or ljq is None:
                    continue

                #if self.K_matrix.file_format == 1:
                omega_matrix[lip,ljq] = 2.* np.sqrt(kss_ip.energy_diff * kss_jq.energy_diff * kss_ip.pop_diff * kss_jq.pop_diff) * self.K_matrix.values[lip,ljq]
                # old Casida format ... doing this already in k_matrix.py
                #elif self.K_matrix.file_format == 0:
                #    omega_matrix[lip,ljq] = self.K_matrix.values[lip,ljq]
                #else:
                #    raise RuntimeError('Invalid K-matrix file format')

                if ( ip == jq ):
                    omega_matrix[lip,ljq] += kss_ip.energy_diff * kss_jq.energy_diff

        # solve eigen system (transpose for scalapack)
        self.eigenvalues = np.zeros(nrows, dtype=float)
        self.eigenvectors = np.zeros((omega_matrix.shape[1],omega_matrix.shape[0]),dtype=float)
        self.eigenvectors[:,:] = np.transpose(omega_matrix)
        layout.diagonalize(self.eigenvectors, self.eigenvalues)
        omega_matrix[:,:] = np.transpose(self.eigenvectors)
        self.eigenvectors = omega_matrix

    def calculate_properties(self):
        if self.custom_axes is not None:
            self.custom_axes = np.array(self.custom_axes)
        #else:
        #    self.custom_axes = np.array([[1.0,0.0,0.0],
        #                                 [0.0,1.0,0.0],
        #                                 [0.0,0.0,1.0]])

        self.lr_comms.parent_comm.barrier()

        # nlrows = self.eigenvectors.shape[0]
        nleigs = self.eigenvectors.shape[1]

        sqrtwloc = np.zeros(nleigs)
        dmxloc  = np.zeros(nleigs)
        dmyloc  = np.zeros(nleigs)
        dmzloc  = np.zeros(nleigs)
        magnxloc = np.zeros(nleigs)
        magnyloc = np.zeros(nleigs)
        magnzloc = np.zeros(nleigs)
        cloc_dm = np.zeros(nleigs)
        cloc_magn = np.zeros(nleigs)

        #print self.lr_comms.parent_comm.rank, self.lr_comms.eh_comm.rank, self.lr_comms.dd_comm.rank, nlrows, nlcols


        # see Autschbach et al., J. Chem. Phys., 116, 6930 (2002)
        # see also Varsano et al., Phys.Chem.Chem.Phys. 11, 4481 (2009)
        #
        # mu_k = sum_jq sqrt(ediff_jq / omega_k)  sqrt(population_jq) * F^(k)_jq * mu_jq
        # m_k  = sum_jq sqrt(omega_k  / ediff_jq) sqrt(population_jq) * F^(k)_jq * m_jq
        #
        # FIXME: check once more
        # mu_k = sum_jq c1_k * c2_k * mu_jq
        #      = sum_jq sqrt(ediff_jq / omega_k) sqrt(fdiff_jq) * F^(k)_jq mu_jq
        # m_k  = sum_jq c2_k / c1_k * m_jq
        #      = sum_jq sqrt(fdiff_jq) sqrt(omega_k / ediff_jq) * F^(k)_jq m_jq
        #
        # c1 =  sqrt(ediff_jq / omega_k)
        #    =  np.sqrt(kss_jq.energy_diff / self.get_excitation_energy(k))
        #
        # c2 = sqrt(fdiff_jq) * F^(k)_jq
        #    = np.sqrt(kss_jq.pop_diff) * self.get_local_eig_coeff(k,self.index_map[(i,p)])
        #

        # debug
        #if 0:
        #    for p in range(self.lr_comms.parent_comm.size):
        #        self.lr_comms.parent_comm.barrier()
        #        if p == self.lr_comms.parent_comm.rank:
        #            print self.lr_comms.parent_comm.rank, self.lr_comms.eh_comm.rank, self.lr_comms.dd_comm.rank
        #            for kloc in range(nleigs):
        #                print self.lr_comms.get_global_dd_index(kloc),
        #            print

        #self.eigenvalues = np.array(range(len(self.eigenvalues)))**4

        # sqrt(omega_kloc)
        for kloc in range(nleigs):
            sqrtwloc[kloc] = np.sqrt( np.sqrt(self.eigenvalues[self.lr_comms.get_global_dd_index(kloc)]) )

        #print self.lr_comms.parent_comm.rank, self.lr_comms.eh_comm.rank, self.lr_comms.dd_comm.rank, nlrows, nleigs, sqrtwloc


        # loop over ks singles
        for (ip, kss_ip) in enumerate(self.ks_singles.kss_list):

            # local ip index
            lip = self.lr_comms.get_local_eh_index(ip)
            #print self.lr_comms.parent_comm.rank, i,p,ip,lip
            if lip is None:
                continue

            # now self.eigenvectors[iploc,kloc]
            # is the "iploc"th local element of
            # the "kloc"th local eigenvector
            #
            # init local mu and m sums with
            # a continuous copy of ROW of eigenvectors
            # i.e. a "ip"th element of each eigenvector
            # (NOT the "jq"th eigenvector)
            cloc_dm[:]   = self.eigenvectors[lip,:]
            cloc_magn[:] = self.eigenvectors[lip,:]

            #print self.lr_comms.parent_comm.rank, i,p, ip, lip, cloc_dm, sqrtwloc, cloc_dm * sqrtwloc

            # c1 * c2 * F = sqrt(fdiff_ip * ediff_ip) / sqrt(omega_k) F^(k)_ip
            cloc_dm   *= np.sqrt(kss_ip.pop_diff * kss_ip.energy_diff)
            cloc_dm   /= sqrtwloc

            # c2 / c1 * F = sqrt(fdiff_ip / ediff_ip) * sqrt(omega_k) F^(k)_ip
            cloc_magn *= np.sqrt(kss_ip.pop_diff / kss_ip.energy_diff)
            cloc_magn *= sqrtwloc

            if self.custom_axes is None:
                dmxloc  += kss_ip.dip_mom_r[0] * cloc_dm
                dmyloc  += kss_ip.dip_mom_r[1] * cloc_dm
                dmzloc  += kss_ip.dip_mom_r[2] * cloc_dm

                magnxloc += kss_ip.magn_mom[0] * cloc_magn
                magnyloc += kss_ip.magn_mom[1] * cloc_magn
                magnzloc += kss_ip.magn_mom[2] * cloc_magn
            else:
                dmxloc += np.dot(kss_ip.dip_mom_r, self.custom_axes[0]) * cloc_dm
                dmyloc += np.dot(kss_ip.dip_mom_r, self.custom_axes[1]) * cloc_dm
                dmzloc += np.dot(kss_ip.dip_mom_r, self.custom_axes[2]) * cloc_dm

                magnxloc += np.dot(kss_ip.magn_mom, self.custom_axes[0]) * cloc_magn
                magnyloc += np.dot(kss_ip.magn_mom, self.custom_axes[1]) * cloc_magn
                magnzloc += np.dot(kss_ip.magn_mom, self.custom_axes[2]) * cloc_magn


        # global properties, but local eigs
        # sum over iploc dmx_iploc[kloc] to dmx[kloc]
        props = np.array([ dmxloc,   dmyloc,   dmzloc,
                           magnxloc, magnyloc, magnzloc ])
        self.lr_comms.eh_comm.sum(props.ravel())
        [ dmxloc,   dmyloc,   dmzloc,
          magnxloc, magnyloc, magnzloc ] = props


        # global properties and eigs
        # create local part of the global eigs
        nrows = len(self.ks_singles.kss_list)
        self.transition_properties = np.zeros([nrows,1+3+3])
        for k in range(nrows):
            kloc = self.lr_comms.get_local_dd_index(k)
            if kloc is None:
                continue
            self.transition_properties[k,0] = sqrtwloc[kloc] * sqrtwloc[kloc]
            self.transition_properties[k,1] = dmxloc[kloc]
            self.transition_properties[k,2] = dmyloc[kloc]
            self.transition_properties[k,3] = dmzloc[kloc]
            self.transition_properties[k,4] = magnxloc[kloc]
            self.transition_properties[k,5] = magnyloc[kloc]
            self.transition_properties[k,6] = magnzloc[kloc]

        # sum over kloc dmx[...kloc...] to dmx[k]
        self.lr_comms.dd_comm.sum(self.transition_properties.ravel())


        #print self.lr_comms.parent_comm.rank, self.transition_properties[:,1]
        #return

        #if self.lr_comms.parent_comm.rank == 0:
        #    print self.transition_properties[:,0]*27.211
        #    print self.transition_properties[:,1]**2 * 2 * self.transition_properties[:,0]
        #    print self.transition_properties[:,2]**2 * 2 * self.transition_properties[:,0]
        #    print self.transition_properties[:,3]**2 * 2 * self.transition_properties[:,0]

        self.trans_prop_ready = True

    # omega_k = sqrt(lambda_k)
    def get_excitation_energy(self, k, units='au'):
        """Get excitation energy for kth interacting transition

        Input parameters:

        k
          Transition index (indexing starts from zero).

        units
          Units for excitation energy: 'au' (Hartree) or 'eV'.
        """
        if not self.trans_prop_ready:
            self.calculate()

        if units == 'au':
            return np.sqrt(self.eigenvalues[k])
        elif units == 'eV' or units == 'eVcgs':
            return np.sqrt(self.eigenvalues[k]) * ase.units.Hartree
        else:
            raise RuntimeError('Unknown units.')


    def get_oscillator_strength(self, k):
        """Get oscillator strength for an interacting transition

        Returns oscillator strength of kth interacting transition.

        Input parameters:

        k
          Transition index (indexing starts from zero).
        """
        if not self.trans_prop_ready:
            self.calculate()

        omega = self.transition_properties[k][0]
        dmx = self.transition_properties[k][1]
        dmy = self.transition_properties[k][2]
        dmz = self.transition_properties[k][3]

        oscx = 2. * omega * dmx * dmx
        oscy = 2. * omega * dmy * dmy
        oscz = 2. * omega * dmz  *dmz
        osc = (oscx + oscy + oscz) / 3.
        return osc, np.array([oscx, oscy, oscz])


    def get_rotatory_strength(self, k, units='au'):
        """Get rotatory strength for an interacting transition

        Returns rotatory strength of kth interacting transition.

        Input parameters:

        k
          Transition index (indexing starts from zero).

        units
          Units for rotatory strength: 'au' or 'cgs'.
        """

        if not self.trans_prop_ready:
            self.calculate()

        # omega = self.transition_properties[k][0]
        dmx = self.transition_properties[k][1]
        dmy = self.transition_properties[k][2]
        dmz = self.transition_properties[k][3]
        magnx = self.transition_properties[k][4]
        magny = self.transition_properties[k][5]
        magnz = self.transition_properties[k][6]

        if units == 'au':
            return - ( dmx * magnx + dmy * magny + dmz * magnz )
        elif units == 'cgs' or units == 'eVcgs':
            # 10^-40 esu cm erg / G
            # = 3.33564095 * 10^-15 A^2 m^3 s
            # conversion factor 471.43 from
            # T. B. Pedersen and A. E. Hansen,
            # Chem. Phys. Lett. 246 (1995) 1
            # From turbomole 64604.8164
            #
            # FIX ME: why?
            # 64604.8164/471.43 = 137.040
            # is the inverse of fine-structure constant
            # OR it must be speed of light in atomic units
            return - 64604.8164 * ( dmx * magnx + dmy * magny + dmz * magnz )
        else:
            raise RuntimeError('Unknown units.')

    def get_transitions(self, filename=None, min_energy=0.0, max_energy=30.0, units='eVcgs'):
        """Get transitions: energy, dipole strength and rotatory strength.

        Returns transitions as (w,S,R, Sx,Sy,Sz) where w is an array of frequencies,
        S is an array of corresponding dipole strengths, and R is an array of
        corresponding rotatory strengths.

        Input parameters:

        min_energy
          Minimum energy

        min_energy
          Maximum energy

        units
          Units for spectrum: 'au' or 'eVcgs'
        """
        if not self.trans_prop_ready:
            self.calculate()

        w = []
        S = []
        R = []
        Sx = []
        Sy = []
        Sz = []

        for (k, omega) in enumerate(self.eigenvalues):
            ww = self.get_excitation_energy(k, units)
            if ww < min_energy or ww > max_energy:
                continue

            w.append(ww)
            St, Sc = self.get_oscillator_strength(k)
            S.append(St)
            Sx.append(Sc[0])
            Sy.append(Sc[1])
            Sz.append(Sc[2])
            R.append(self.get_rotatory_strength(k,units))

        w = np.array(w)
        S = np.array(S)
        R = np.array(R)
        Sx = np.array(Sx)
        Sy = np.array(Sy)
        Sz = np.array(Sz)


        if filename is not None and gpaw.mpi.world.rank == 0:
            sfile = open(filename,'w')
            sfile.write("# %12s  %12s  %12s     %12s  %12s  %12s    %s\n" % ('energy','osc str','rot str','osc str x', 'osc str y', 'osc str z', 'units: ' + units))
            for (ww,SS,RR,SSx,SSy,SSz) in zip(w,S,R,Sx,Sy,Sz):
                sfile.write("  %12.8lf  %12.8lf  %12.8lf     %12.8lf  %12.8lf  %12.8lf\n" % (ww,SS,RR,SSx,SSy,SSz))
            sfile.close()

        return (w,S,R,Sx,Sy,Sz)

    def get_spectrum(self, filename=None, min_energy=0.0, max_energy=30.0,
                     energy_step=0.01, width=0.1, units='eVcgs'):
        """Get spectrum for dipole and rotatory strength.

        Returns folded spectrum as (w,S,R) where w is an array of frequencies,
        S is an array of corresponding dipole strengths, and R is an array of
        corresponding rotatory strengths.

        Input parameters:

        min_energy
          Minimum energy

        min_energy
          Maximum energy

        energy_step
          Spacing between calculated energies

        width
          Width of the Gaussian

        units
          Units for spectrum: 'au' or 'eVcgs'
        """

        if not self.trans_prop_ready:
            self.calculate()

        (ww, SS, RR, SSx, SSy, SSz) = self.get_transitions(min_energy=min_energy-5*width,
                                                           max_energy=max_energy+5*width,
                                                           units=units)

        if units == 'eVcgs':
            pass  # convf = 1/ase.units.Hartree
        elif units == 'au':
            assert 0  # convf = 1.
        else:
            raise RuntimeError('Invalid units')

        w = np.arange(min_energy, max_energy, energy_step)
        S  = np.zeros_like(w)
        Sx = np.zeros_like(w)
        Sy = np.zeros_like(w)
        Sz = np.zeros_like(w)
        R  = np.zeros_like(w)


        for (k, www) in enumerate(ww):

            c = SS[k] / width / np.sqrt(2*np.pi)
            S += c * np.exp( (-.5/width/width) * np.power(w-ww[k],2) )

            c = SSx[k] / width / np.sqrt(2*np.pi)
            Sx += c * np.exp( (-.5/width/width) * np.power(w-ww[k],2) )

            c = SSy[k] / width / np.sqrt(2*np.pi)
            Sy += c * np.exp( (-.5/width/width) * np.power(w-ww[k],2) )

            c = SSz[k] / width / np.sqrt(2*np.pi)
            Sz += c * np.exp( (-.5/width/width) * np.power(w-ww[k],2) )


            c = RR[k] / width / np.sqrt(2*np.pi)
            R += c * np.exp( (-.5/width/width) * np.power(w-ww[k],2) )


        if filename is not None and gpaw.mpi.world.rank == 0:
            sfile = open(filename,'w')
            sfile.write("# %12s  %12s  %12s     %12s  %12s  %12s    %s\n" % ('energy','osc str','rot str','osc str x','osc str y','osc str z', 'units: ' + units))
            for (ww,SS,RR,SSx,SSy,SSz) in zip(w,S,R,Sx,Sy,Sz):
                sfile.write("  %12.8lf  %12.8lf  %12.8lf     %12.8lf  %12.8lf  %12.8lf\n" % (ww,SS,RR,SSx,SSy,SSz))
            sfile.close()

        return (w,S,R, Sx,Sy,Sz)



    def get_transition_contributions(self, index_of_transition):
        """Get contributions of Kohn-Sham singles to a given transition.

        Includes population difference.

        For large systems this is slow.

        Input parameters:

        index_of_transition:
          index of transition starting from zero
        """

        neigs = len(self.ks_singles.kss_list)
        f2 = np.zeros(neigs)

        if index_of_transition < 0:
            raise RuntimeError('Error in get_transition_contributions: Index < zero.')
        if index_of_transition >= neigs:
            raise RuntimeError('Error in get_transition_contributions: Index >= number of transitions')

        k = index_of_transition
        # local k index
        kloc = self.lr_comms.get_local_dd_index(k)

        if kloc is not None:

            for (ip,kss_ip) in enumerate(self.ks_singles.kss_list):
                # local ip index
                lip = self.lr_comms.get_local_eh_index(ip)
                if lip is None:
                    continue

                # self.eigenvectors[iploc,kloc]
                # is the "iploc"th local element of
                # the "kloc"th local eigenvector
                f2[ip] = self.eigenvectors[lip,kloc] * self.eigenvectors[lip,kloc] * kss_ip.pop_diff

        self.lr_comms.parent_comm.sum(f2)

        return f2

