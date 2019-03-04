import os
import glob
import datetime

import numpy as np

import gpaw.mpi
from gpaw.utilities import pack
from gpaw.lrtddft2.eta import QuadraticETA


class Kmatrix:
    def __init__(self, ks_singles, xc, deriv_scale=1e-5):
        self.basefilename = ks_singles.basefilename
        self.ks_singles = ks_singles
        self.lr_comms = self.ks_singles.lr_comms
        self.calc = self.ks_singles.calc
        self.xc = xc
        self.ready_indices = []
        self.values = None
        self.K_matrix_ready = False
        self.deriv_scale = deriv_scale
        self.file_format = 0 # 0 = 'Casida', 1 = 'K-matrix'

        self.dVxct_gip_2 = None       # temporary for finite difference

        # for pros!!!
        self.fH_pre = 1.0
        self.fxc_pre = 1.0

    def initialize(self):
        self.K_matrix_ready = False

    def read_indices(self):
        # Read ALL ready_rows files
        #self.timer.start('Init read ready rows')
        # root reads and then broadcasts
        data = None
        if self.lr_comms.parent_comm.rank == 0:
            data = ''
            ready_files = glob.glob(self.basefilename+'.ready_rows.*')
            for ready_file in ready_files:
                if os.path.isfile(ready_file):
                    data += open(ready_file,'r',1024*1024).read()

        data = gpaw.mpi.broadcast_string(data, root=0, comm=self.lr_comms.parent_comm)
        for line in data.splitlines():
            line = line.split()
            self.ready_indices.append([int(line[0]),int(line[1])])

    def read_values(self):
        """ Read K-matrix (not the Casida form) ready for 2D ScaLapack array"""
        # nrow = len(self.ks_singles.kss_list)  # total rows
        nlrow = 0 # local rows
        nlcol = 0 # local cols

        # Create indexing
        self.lr_comms.index_map = {}        # (i,p) to matrix index map
        for (ip, kss) in enumerate(self.ks_singles.kss_list):
            self.lr_comms.index_map[(kss.occ_ind,kss.unocc_ind)] = ip
            if self.lr_comms.get_local_eh_index(ip) is not None:
                nlrow += 1
            if self.lr_comms.get_local_dd_index(ip) is not None:
                nlcol += 1

        # ready rows list for different procs (read by this proc)
        elem_lists = {}
        for proc in range(self.lr_comms.parent_comm.size):
           elem_lists[proc] = []

        self.lr_comms.parent_comm.barrier()

        #self.timer.start('Read K-matrix')
        # Read ALL ready_rows files but on different processors
        for (k, K_fn) in enumerate(glob.glob(self.basefilename + '.K_matrix.*')):
            # read every "parent comm size"th file, starting from parent comm rank
            if k % self.lr_comms.parent_comm.size != self.lr_comms.parent_comm.rank:
                continue


            # for each file
            for line in open(K_fn, 'r', 1024 * 1024).read().splitlines():
                #self.timer.start('Read K-matrix: elem')
                if line[0] == '#':
                    if line.startswith('# K-matrix file'):
                        self.file_format = 1
                    continue

                elems = line.split()
                i = int(elems[0])
                p = int(elems[1])
                j = int(elems[2])
                q = int(elems[3])
                #self.timer.stop('Read K-matrix: elem')

                #self.timer.start('Read K-matrix: index')
                ip = self.lr_comms.index_map.get( (i,p) )
                jq = self.lr_comms.index_map.get( (j,q) )
                #self.timer.stop('Read K-matrix: index')
                if ip is None or jq is None:
                    continue

                # where to send
                #self.timer.start('Read K-matrix: line')
                (proc, ehproc, ddproc, lip, ljq) = self.lr_comms.get_matrix_elem_proc_and_index(ip, jq)
                elem_lists[proc].append( line + '\n')
                #self.timer.stop('Read K-matrix: line')

                if ip == jq: continue

                # where to send transposed
                #self.timer.start('Read K-matrix: line')
                (proc, ehproc, ddproc, lip, ljq) = self.lr_comms.get_matrix_elem_proc_and_index(jq, ip)
                elem_lists[proc].append( line + '\n')
                #self.timer.stop('Read K-matrix: line')


        #if self.parent_comm.rank == 0:
        #    print '-------------- READING K-MATRIX done --------------------', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        #    print 'elem_list sizes:',
        #    for proc in range(self.parent_comm.size):
        #        print len(elem_lists[proc]),
        #    print


        #self.timer.start('Read K-matrix: join')
        for proc in range(self.lr_comms.parent_comm.size):
            elem_lists[proc] = ''.join(elem_lists[proc])
        #self.timer.stop('Read K-matrix: join')

        #print self.parent_comm.rank, '- elem_lists -'
        #for (key,val) in elem_lists.items():
        #    print key, ':', val[0:120]
        #sys.stdout.flush()

        self.lr_comms.parent_comm.barrier()
        #self.timer.stop('Read K-matrix')

        self.file_format = self.lr_comms.parent_comm.max(self.file_format)

        # send and receive elem_list
        #self.timer.start('Communicate K-matrix')
        alltoall_dict = gpaw.mpi.alltoallv_string(elem_lists, self.lr_comms.parent_comm)
        # ready for garbage collection
        del elem_lists
        local_elem_list = ''.join(alltoall_dict.values())
        # ready for garbage collection
        del alltoall_dict



        #if self.parent_comm.rank == 0:
        #    print '-------------- communicating K-MATRIX done --------------------', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        #    print 'local elem list size: ', len(local_elem_list)


        #local_elem_list = ''
        #for sending_proc in range(self.parent_comm.size):
        #    for receiving_proc in range(self.parent_comm.size):
        #        if ( sending_proc == receiving_proc and
        #             sending_proc == self.parent_comm.rank ):
        #            local_elem_list += elem_lists[sending_proc]
        #        elif sending_proc == self.parent_comm.rank:
        #            gpaw.mpi.send_string( elem_lists[receiving_proc],
        #                                  receiving_proc,
        #                                  comm=self.parent_comm)
        #        elif receiving_proc == self.parent_comm.rank:
        #            elist = gpaw.mpi.receive_string( sending_proc,
        #                                             comm=self.parent_comm )
        #            local_elem_list += elist
        #


        #print self.parent_comm.rank, '- local_elem_list -', local_elem_list[0:120].replace('\n', ' | ')
        #sys.stdout.flush()
        #sys.exit(0)

        #self.timer.stop('Communicate K-matrix')

        #self.timer.start('Build matrix')

        # Matrix build
        K_matrix = np.zeros((nlrow,nlcol))
        K_matrix[:,:] = np.NAN # fill with NaNs to detect problems
        # Read ALL K_matrix files
        for line in local_elem_list.splitlines():
            line = line.split()
            ipkey = (int(line[0]), int(line[1]))
            jqkey = (int(line[2]), int(line[3]))
            Kvalue = float(line[4])

            ip = self.lr_comms.index_map.get(ipkey)
            jq = self.lr_comms.index_map.get(jqkey)

            # if not in index map, ignore
            if ( ip is None or jq is None ):
                continue

            kss_ip = self.ks_singles.kss_list[ip]
            kss_jq = self.ks_singles.kss_list[jq]

            # if (ip,jq) on this this proc
            lip = self.lr_comms.get_local_eh_index(ip)
            ljq = self.lr_comms.get_local_dd_index(jq)
            if lip is not None and ljq is not None:
                # add value to matrix
                if self.file_format == 1:
                    K_matrix[lip,ljq] = Kvalue
                elif self.file_format == 0:
                    K_matrix[lip,ljq] = Kvalue / ( 2.* np.sqrt(kss_ip.energy_diff * kss_jq.energy_diff * kss_ip.pop_diff * kss_jq.pop_diff) )
                else:
                    raise RuntimeError('Invalid K-matrix file format')

            # if (jq,ip) on this this proc
            ljq = self.lr_comms.get_local_eh_index(jq)
            lip = self.lr_comms.get_local_dd_index(ip)
            if lip is not None and ljq is not None:
                # add value to matrix
                if self.file_format == 1:
                    K_matrix[ljq,lip] = Kvalue
                elif self.file_format == 0:
                    K_matrix[ljq,lip] = Kvalue / ( 2.* np.sqrt(kss_ip.energy_diff * kss_jq.energy_diff * kss_ip.pop_diff * kss_jq.pop_diff) )
                else:
                    raise RuntimeError('Invalid K-matrix file format')

        # ready for garbage collection
        del local_elem_list

        #self.timer.stop('Build matrix')

        #if self.parent_comm.rank == 0:
        #    print '-------------- building K-MATRIX done --------------------', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


        #print K_matrix


        #for ((i,p), ip) in self.index_map.items():
        #    for ((j,q), jq) in self.index_map.items():
        #        lip = self.get_local_eh_index(ip)
        #        ljq = self.get_local_dd_index(jq)
        #        if ( lip is not None and ljq is not None ):
        #            print 'proc #', self.parent_comm.rank, ' dd #', self.dd_comm.rank, ' eh #', self.eh_comm.rank, ': ',ip,'(', i, ',', p,') ',jq,'(', j, ',', q,')  = ', K_matrix[lip,ljq]


        # If any NaNs found, we did not read all matrix elements... BAD
        if np.isnan(np.sum(np.sum(K_matrix))):
            raise RuntimeError('Not all required K-matrix elements could be found.')

        self.values = K_matrix

    # FIXME: implement spin polarized
    def calculate(self):
        # Check if already done before allocating
        if self.K_matrix_ready:
            return

        # Loop over all transitions
        self.K_matrix_ready = True  # mark done... if not, it's changed
        nrows = 0                   # number of rows for timings
        for (ip,kss_ip) in enumerate(self.ks_singles.kss_list):
            i = kss_ip.occ_ind
            p = kss_ip.unocc_ind

            # if not mine, skip it
            if self.lr_comms.get_local_eh_index(ip) is None:
                continue
            # if already calculated, skip it
            if [i,p] in self.ready_indices:
                continue

            self.K_matrix_ready = False  # something not calculated, must do it
            nrows += 1

        # If matrix was ready, done
        if self.K_matrix_ready:
            return


        #self.timer.start('Calculate K matrix')
        #self.timer.start('Initialize')

        # Filenames
        #######################################################################
        # saving CORRECT K-matrix, not its "Casida form" like in alpha version
        #######################################################################
        Kfn = self.basefilename + '.K_matrix.' + '%06dof%06d' % (self.lr_comms.eh_comm.rank, self.lr_comms.eh_comm.size)
        rrfn = self.basefilename + '.ready_rows.' + '%06dof%06d' % (self.lr_comms.eh_comm.rank, self.lr_comms.eh_comm.size)
        logfn = self.basefilename + '.log.' + '%06dof%06d' % (self.lr_comms.eh_comm.rank, self.lr_comms.eh_comm.size)

        # Open only on dd_comm root
        if self.lr_comms.dd_comm.rank == 0:
            self.Kfile = open(Kfn, 'a+')
            self.ready_file = open(rrfn,'a+')
            self.log_file = open(logfn,'a+')
            self.Kfile.write('# K-matrix file\n')

        self.poisson = self.calc.hamiltonian.poisson

        # Allocate grids for densities and potentials
        dnt_Gip = self.calc.wfs.gd.empty()
        dnt_gip = self.calc.density.finegd.empty()
        drhot_gip = self.calc.density.finegd.empty()
        dnt_Gjq = self.calc.wfs.gd.empty()
        dnt_gjq = self.calc.density.finegd.empty()
        drhot_gjq = self.calc.density.finegd.empty()

        self.nt_g = self.calc.density.finegd.zeros(self.calc.density.nt_sg.shape[0])

        dVht_gip = self.calc.density.finegd.empty()
        dVxct_gip = self.calc.density.finegd.zeros(self.calc.density.nt_sg.shape[0])


        #self.timer.stop('Initialize')
        # Init ETA
        self.matrix_eta = QuadraticETA()

        #################################################################
        # Outer loop over KS singles
        for (ip,kss_ip) in enumerate(self.ks_singles.kss_list):

            # if not mine, skip it
            if self.lr_comms.get_local_eh_index(ip) is None:
                continue
            # if already calculated, skip it
            if [kss_ip.occ_ind,kss_ip.unocc_ind] in self.ready_indices:
                continue

            # ETA
            if self.lr_comms.dd_comm.rank == 0:
                self.matrix_eta.update()
                # add 21% extra time (1.1**2 = 1.21)
                eta = self.matrix_eta.eta( (nrows+1) * 1.1)
                self.log_file.write('Calculating pair %5d => %5d  ( %s, ETA %9.1lfs )\n' % (kss_ip.occ_ind, kss_ip.unocc_ind, str(datetime.datetime.now()), eta))
                self.log_file.flush()


            # Pair density
            #self.timer.start('Pair density')
            dnt_Gip[:] = 0.0
            dnt_gip[:] = 0.0
            drhot_gip[:] = 0.0
            kss_ip.calculate_pair_density(dnt_Gip, dnt_gip, drhot_gip)
            #self.timer.stop('Pair density')


            # Smooth Hartree "pair" potential
            # for compensated pair density drhot_gip
            #self.timer.start('Poisson')
            dVht_gip[:] = 0.0
            self.poisson.solve(dVht_gip, drhot_gip, charge=None)
            #self.timer.stop('Poisson')

            # smooth XC "pair" potential
            self.calculate_smooth_xc_pair_potential(kss_ip, dnt_gip, dVxct_gip)
            # paw term of XC "pair" potential
            I_asp = self.calculate_paw_xc_pair_potentials(kss_ip)


            #################################################################
            # Inner loop over KS singles
            K = [] # storage for row before writing to file
            for (jq,kss_jq) in enumerate(self.ks_singles.kss_list):
                i = kss_ip.occ_ind
                p = kss_ip.unocc_ind
                j = kss_jq.occ_ind
                q = kss_jq.unocc_ind

                # Only lower triangle
                if ip < jq: continue

                # Pair density dn_jq
                #self.timer.start('Pair density')
                dnt_Gjq[:] = 0.0
                dnt_gjq[:] = 0.0
                drhot_gjq[:] = 0.0
                kss_jq.calculate_pair_density(dnt_Gjq, dnt_gjq, drhot_gjq)
                #self.timer.stop('Pair density')


                # integrate to get the final matrix element value
                #self.timer.start('Integrate')

                # init grid part
                Ig = 0.0

                # Hartree smooth part, RHOT_JQ HERE???
                Ig += self.fH_pre * self.calc.density.finegd.integrate(dVht_gip, drhot_gjq)
                # XC smooth part
                Ig += self.fxc_pre * self.calc.density.finegd.integrate(dVxct_gip, dnt_gjq)
                #self.timer.stop('Integrate')


                # Atomic corrections
                #self.timer.start('Atomic corrections')
                Ia = self.calculate_paw_fHXC_corrections(kss_ip, kss_jq, I_asp)

                #self.timer.stop('Atomic corrections')
                Ia = self.lr_comms.dd_comm.sum(Ia)

                # Total integral
                Itot = Ig + Ia


                # K_ip,jq += <ip|fHxc|jq>
                K.append( [i,p,j,q, Itot] )

            # Write  i p j q Kipjq
            # (format: -2345.789012345678)

            #self.timer.start('Write K')
            # Write only on dd_comm root
            if self.lr_comms.dd_comm.rank == 0:

                # Write only lower triangle of K-matrix
                for [i,p,j,q,Kipjq] in K:
                    self.Kfile.write("%5d %5d %5d %5d %22.16lf\n" % (i,p,j,q,Kipjq))
                self.Kfile.flush() # flush K-matrix before ready_rows

                # Write and flush ready rows
                self.ready_file.write("%d %d\n" % (kss_ip.occ_ind,
                                                   kss_ip.unocc_ind))
                self.ready_file.flush()

            #self.timer.stop('Write K')

            # Update ready rows before continuing
            self.ready_indices.append([kss_ip.occ_ind,kss_ip.unocc_ind])



        # Close files on dd_comm root
        if self.lr_comms.dd_comm.rank == 0:
            self.Kfile.close()
            self.ready_file.close()
            self.log_file.close()

        #self.timer.stop('Calculate K matrix')


    def calculate_smooth_xc_pair_potential(self, kss_ip, dnt_gip, dVxct_gip):
        # Smooth xc potential
        # (finite difference approximation from xc-potential)
        if ( self.dVxct_gip_2 is None ):
            self.dVxct_gip_2 = self.calc.density.finegd.zeros(self.calc.density.nt_sg.shape[0])
        dVxct_gip_2 = self.dVxct_gip_2

        s = kss_ip.spin_ind

        #self.timer.start('Smooth XC')
        # finite difference plus,  vxc+ = vxc(n + deriv_scale * dn)
        self.nt_g[s][:] = self.deriv_scale * dnt_gip
        self.nt_g[s][:] += self.calc.density.nt_sg[s]
        dVxct_gip[:] = 0.0
        self.xc.calculate(self.calc.density.finegd, self.nt_g, dVxct_gip)

        # finite difference minus, vxc+ = vxc(n - deriv_scale * dn)
        self.nt_g[s][:] = -self.deriv_scale * dnt_gip
        self.nt_g[s][:] += self.calc.density.nt_sg[s]
        dVxct_gip_2[:] = 0.0
        self.xc.calculate(self.calc.density.finegd, self.nt_g, dVxct_gip_2)
        dVxct_gip -= dVxct_gip_2
        # finite difference approx for fxc
        # vxc = (vxc+ - vxc-) / 2h
        dVxct_gip *= 1./(2.*self.deriv_scale)
        #self.timer.stop('Smooth XC')


    def calculate_paw_xc_pair_potentials(self, kss_ip):
        # XC corrections
        I_asp = {}
        i = kss_ip.occ_ind
        p = kss_ip.unocc_ind

        # s = kss_ip.spin_ind

        # FIXME, only spin unpolarized works
        #self.timer.start('Atomic XC')
        for a, P_ni in self.calc.wfs.kpt_u[kss_ip.kpt_ind].P_ani.items():
            I_sp = np.zeros_like(self.calc.density.D_asp[a])
            I_sp_2 = np.zeros_like(self.calc.density.D_asp[a])

            Pip_ni = self.calc.wfs.kpt_u[kss_ip.spin_ind].P_ani[a]
            Dip_ii = np.outer(Pip_ni[i], Pip_ni[p])
            Dip_p  = pack(Dip_ii)

            # finite difference plus
            D_sp = self.calc.density.D_asp[a].copy()
            D_sp[kss_ip.spin_ind] += self.deriv_scale * Dip_p
            self.xc.calculate_paw_correction(self.calc.wfs.setups[a],
                                             D_sp, I_sp)

            # finite difference minus
            D_sp_2 = self.calc.density.D_asp[a].copy()
            D_sp_2[kss_ip.spin_ind] -= self.deriv_scale * Dip_p
            self.xc.calculate_paw_correction(self.calc.wfs.setups[a],
                                             D_sp_2, I_sp_2)

            # finite difference
            I_asp[a] = (I_sp - I_sp_2) / (2.*self.deriv_scale)

        #self.timer.stop('Atomic XC')

        return I_asp


    def calculate_paw_fHXC_corrections(self, kss_ip, kss_jq, I_asp):
        i = kss_ip.occ_ind
        p = kss_ip.unocc_ind
        j = kss_jq.occ_ind
        q = kss_jq.unocc_ind

        Ia = 0.0
        for a, P_ni in self.calc.wfs.kpt_u[kss_jq.spin_ind].P_ani.items():
            Pip_ni = self.calc.wfs.kpt_u[kss_ip.spin_ind].P_ani[a]
            Dip_ii = np.outer(Pip_ni[i], Pip_ni[p])
            Dip_p = pack(Dip_ii)

            Pjq_ni = self.calc.wfs.kpt_u[kss_jq.spin_ind].P_ani[a]
            Djq_ii = np.outer(Pjq_ni[j], Pjq_ni[q])
            Djq_p = pack(Djq_ii)

            # Hartree part
            C_pp = self.calc.wfs.setups[a].M_pp
            # why factor of two here?
            # see appendix A of J. Chem. Phys. 128, 244101 (2008)
            #
            # 2 sum_prst      P   P   C     P   P
            #                  ip  jr  prst  ks  qt
            Ia += self.fH_pre * 2.0 * np.dot(Djq_p, np.dot(C_pp, Dip_p))

            # XC part, CHECK THIS JQ EVERWHERE!!!
            Ia += self.fxc_pre * np.dot(I_asp[a][kss_jq.spin_ind], Djq_p)

        return Ia
