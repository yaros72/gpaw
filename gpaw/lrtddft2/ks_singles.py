import os

import io

import numpy as np

import ase.units

import gpaw.mpi

from gpaw.utilities.tools import coordinates
from gpaw.utilities.tools import pick
from gpaw.utilities import pack

from gpaw.fd_operators import Gradient


class KohnShamSingleExcitation:
    def __init__(self, gs_calc, kpt_ind, occ_ind, unocc_ind):
        self.calc = gs_calc
        self.kpt_ind = kpt_ind
        self.spin_ind = self.calc.wfs.kpt_u[self.kpt_ind].s
        self.occ_ind = occ_ind
        self.unocc_ind = unocc_ind
        self.energy_diff = None
        self.pop_diff = None
        self.dip_mom_r = None
        self.magn_mom = None
        self.pair_density = None

    # Calculates pair density of (i,p) transition (and given kpt)
    # dnt_Gp: pair density without compensation charges on coarse grid
    # dnt_gp: pair density without compensation charges on fine grid      (XC)
    # drhot_gp: pair density with compensation charges on fine grid  (poisson)
    def calculate_pair_density(self, dnt_Gip, dnt_gip, drhot_gip):
        if self.pair_density is None:
            # FIXME: USE EXISTING PairDensity method
            self.pair_density = LRiPairDensity(self.calc.density)
        self.pair_density.initialize(self.calc.wfs.kpt_u[self.kpt_ind],
                                     self.occ_ind, self.unocc_ind)
        self.pair_density.get(dnt_Gip, dnt_gip, drhot_gip)

    def __str__(self):
        if self.dip_mom_r is not None and self.dip_mom_v is not None and self.magn_mom is not None:
            str = "# KS single excitation from state %05d to state %05d: dE_pi = %18.12lf, f_pi = %18.12lf,  dmr_ip = (%18.12lf, %18.12lf, %18.12lf), dmv_ip = (%18.12lf, %18.12lf, %18.12lf), dmm_ip = %18.12lf" \
                % ( self.occ_ind, \
                        self.unocc_ind, \
                        self.energy_diff, \
                        self.pop_diff, \
                        self.dip_mom_r[0], self.dip_mom_r[1], self.dip_mom_r[2], \
                        self.dip_mom_v[0], self.dip_mom_v[1], self.dip_mom_v[2], \
                        self.magn_mom )
        elif self.energy_diff is not None and self.pop_diff is not None:
            str = "# KS single excitation from state %05d to state %05d: dE_pi = %18.12lf, f_pi = %18.12lf" \
                % ( self.occ_ind, \
                        self.unocc_ind,  \
                        self.energy_diff, \
                        self.pop_diff )
        elif self.occ_ind is not None and self.unocc_ind is not None:
            str = "# KS single excitation from state %05d to state %05d" \
                % ( self.occ_ind, self.unocc_ind )
        else:
            raise RuntimeError("Uninitialized KSSingle")
        return str


class KohnShamSingles:
    def __init__(self,
                 basefilename,
                 gs_calc,
                 kpt_index,
                 min_occ, max_occ,
                 min_unocc, max_unocc,
                 max_energy_diff,
                 min_pop_diff,
                 lr_comms,
                 txt):
        self.basefilename = basefilename
        self.calc = gs_calc
        self.min_occ = min_occ
        self.max_occ = max_occ
        self.min_unocc = min_unocc
        self.max_unocc = max_unocc
        self.max_energy_diff = max_energy_diff
        self.min_pop_diff = min_pop_diff
        self.kpt_ind = kpt_index
        self.lr_comms = lr_comms
        self.txt = txt

        self.kss_list = None
        self.kss_list_ready = False

        self.kss_prop = None
        self.kss_prop_ready = False

    def update_list(self):
        # shorthands
        eps_n = self.calc.wfs.kpt_u[self.kpt_ind].eps_n      # eigen energies
        f_n = self.calc.wfs.kpt_u[self.kpt_ind].f_n          # occupations

        # Create Kohn-Sham single excitation list with energy filter
        old_kss_list = self.kss_list   # save old list for later
        self.kss_list = []             # create a completely new list
        # Occupied loop
        for i in range(self.min_occ, self.max_occ+1):
            # Unoccupied loop
            for p in range(self.min_unocc, self.max_unocc+1):
                deps_pi = eps_n[p] - eps_n[i] # energy diff
                df_ip = f_n[i] - f_n[p]       # population diff
                # Filter it
                if ( np.abs(deps_pi) <= self.max_energy_diff
                     and df_ip > self.min_pop_diff ):
                    # i, p, deps, df, mur, muv, magn
                    kss = KohnShamSingleExcitation(self.calc, self.kpt_ind, i, p)
                    kss.energy_diff = deps_pi
                    kss.pop_diff = df_ip
                    self.kss_list.append(kss)

        # Sort by energy diff:
        self.kss_list = sorted(self.kss_list, key=lambda x: x.energy_diff)


        # Remove old transitions and add new
        # BUT only add to the end of the list (otherwise lower triangle
        # matrix is not filled completely)
        if old_kss_list is not None:
            new_kss_list = self.kss_list   # required list
            self.kss_list = []             # final list with correct order
            # Old list first
            # If in new and old lists
            for kss_o in old_kss_list:
                found = False
                for kss_n in new_kss_list:
                    if ( kss_o.occ_ind == kss_n.occ_ind
                         and kss_o.unocc_ind == kss_n.unocc_ind ):
                        found = True
                        break
                if found:
                    self.kss_list.append(kss_o) # Found, add to final list
                else:
                    pass                        # else drop


            # Now old transitions which are not in new list where dropped

            # If only in new list
            app_list = []
            for kss_n in new_kss_list:
                found = False
                for kss in self.kss_list:
                    if ( kss.occ_ind == kss_n.occ_ind
                         and kss.unocc_ind == kss_n.unocc_ind ):
                        found = True
                        break
                if not found:
                    app_list.append(kss_n) # Not found, add to final list
                else:
                    pass                   # else skip to avoid duplicates

            # Create the final list
            self.kss_list += app_list


        self.txt.write('Number of electron-hole pairs: %d\n' % len(self.kss_list))
        self.txt.write('Maximum energy difference:     %6.3lf\n' % (self.kss_list[-1].energy_diff * ase.units.Hartree))

        # Prevent repeated work
        self.kss_list_ready = True

    def read(self):
        # Read KS_singles file if exists
        # occ_index | unocc index | energy diff | population diff |
        #   dmx | dmy | dmz | magnx | magny | magnz
        kss_file = self.basefilename + '.KS_singles'
        if os.path.exists(kss_file) and os.path.isfile(kss_file):
            self.kss_list = []

            # root reads and broadcasts
            # a bit ugly and slow but works
            data = None
            if self.lr_comms.parent_comm.rank == 0:
                data = open(kss_file, 'r', 1024*1024).read()
            data = gpaw.mpi.broadcast_string(data, root=0,
                                             comm=self.lr_comms.parent_comm)

            # just parse
            for line in io.StringIO(data):
                line = line.split()
                i, p = int(line[0]), int(line[1])
                ediff, fdiff = float(line[2]), float(line[3])
                dm = np.array([float(line[4]),float(line[5]),float(line[6])])
                mm = np.array([float(line[7]),float(line[8]),float(line[9])])
                kss = KohnShamSingleExcitation(self.calc, self.kpt_ind, i, p)
                kss.energy_diff = ediff
                kss.pop_diff = fdiff
                kss.dip_mom_r = dm
                kss.magn_mom = mm
                #assert self.index_of_kss(i,p) is None, 'KS transition %d->%d found twice in KS_singles files.' % (i,p)
                self.kss_list.append(kss)

            # if none read
            if len(self.kss_list) <= 0:
                self.kss_list = None

    # Calculate dipole moment and magnetic moment for noninteracting
    # Kohn-Sham transitions
    # FIXME: Do it in parallel, currently doing repeated work
    #        Should distribute kss_list over eh_comm
    def calculate(self):
        # Check that kss_list is up-to-date
        if not self.kss_list_ready:
            self.update_list()
            self.kss_prop_ready = False

        # Check if we already have properties for all singles
        if self.kss_prop_ready:
            return
        self.kss_prop_ready = True
        for kss_ip in self.kss_list:
            if kss_ip.dip_mom_r is None or kss_ip.magn_mom is None:
                self.kss_prop_ready = False
                break
        # If had, done
        if self.kss_prop_ready:
            return

        #self.timer.start('Calculate KS properties')


        # Init pair densities
        dnt_Gip = self.calc.wfs.gd.empty()
        dnt_gip = self.calc.density.finegd.empty()
        drhot_gip = self.calc.density.finegd.empty()


        # Init gradients of wfs
        grad_psit2_G = [self.calc.wfs.gd.empty(), self.calc.wfs.gd.empty(),
                        self.calc.wfs.gd.empty()]


        # Init gradient operators
        grad = []
        dtype = pick(self.calc.wfs.kpt_u[self.kpt_ind].psit_nG, 0).dtype
        for c in range(3):
            grad.append(Gradient(self.calc.wfs.gd, c, dtype=dtype, n=2))


        # Coordinate vector r
        R0 = 0.5 * np.diag(self.calc.wfs.gd.cell_cv)
        # R0 = np.array([0.0,0.0,0.0]) # old_version
        #print 'R0', R0
        r_cG, r2_G = coordinates(self.calc.wfs.gd, origin=R0)
        r_cg, r2_g = coordinates(self.calc.density.finegd, origin=R0)


        # Loop over all KS single excitations
        #
        # Transition dipole moment, mu_ip = <p| (-e r) |i>
        # Magnetic transition dipole, m_ip = -(1/2c) <i|L|p>
        # For total m_0I = -m_I0 = -(m_0I)^*, but not for m_ip (right?)
        # R_0I = Im[mu_0I * m_0I]
        # mu_ip^0I =  omega_0I^(-1/2) D_ip      S^(-1/2) F_0I
        # m_ip^0I  = -omega_0I^(+1/2) M_ip C^-1 S^(+1/2) F_0I
        #
        # S_ip,ip = - (eps_p - eps_i) / (n_p - n_i)    (note: n_p < n_i)
        # C_ip,ip = 1 / (n_p - n_i)
        #
        # See:
        # WIREs Comput Mol Sci 2012, 2: 150-166 doi: 10.1002/wcms.55
        # J. Chem. Phys., Vol. 116, 6930 (2002)
        for kss_ip in self.kss_list:
            # If have dipole moment and magnetic moment, already done and skip
            if (kss_ip.dip_mom_r is not None and
                kss_ip.magn_mom is not None):
                continue

            # Transition dipole moment, mu_ip = <p| (-r) |i>
            kss_ip.calculate_pair_density( dnt_Gip, dnt_gip, drhot_gip )
            #kss_ip.dip_mom_r = self.calc.density.finegd.calculate_dipole_moment(drhot_gip)
            #kss_ip.dip_mom_r = self.calc.density.finegd.calculate_dipole_moment(drhot_gip)
            kss_ip.dip_mom_r = np.zeros(3)
            kss_ip.dip_mom_r[0] = -self.calc.density.finegd.integrate(r_cg[0] * drhot_gip)
            kss_ip.dip_mom_r[1] = -self.calc.density.finegd.integrate(r_cg[1] * drhot_gip)
            kss_ip.dip_mom_r[2] = -self.calc.density.finegd.integrate(r_cg[2] * drhot_gip)


            # Magnetic transition dipole, m_ip = -(1/2c) <i|L|p> = i/2c <i|r x p|p>
            # see Autschbach et al., J. Chem. Phys., 116, 6930 (2002)

            # Gradients
            for c in range(3):
                grad[c].apply(kss_ip.pair_density.psit2_G, grad_psit2_G[c], self.calc.wfs.kpt_u[self.kpt_ind].phase_cd)

            # <psi1|r x grad|psi2>
            #    i  j  k
            #    x  y  z   = (y pz - z py)i + (z px - x pz)j + (x py - y px)
            #    px py pz
            rxnabla_g = np.zeros(3)
            rxnabla_g[0] = self.calc.wfs.gd.integrate(kss_ip.pair_density.psit1_G *
                                                      (r_cG[1] * grad_psit2_G[2] -
                                                       r_cG[2] * grad_psit2_G[1]))
            rxnabla_g[1] = self.calc.wfs.gd.integrate(kss_ip.pair_density.psit1_G *
                                                      (r_cG[2] * grad_psit2_G[0] -
                                                       r_cG[0] * grad_psit2_G[2]))
            rxnabla_g[2] = self.calc.wfs.gd.integrate(kss_ip.pair_density.psit1_G *
                                                      (r_cG[0] * grad_psit2_G[1] -
                                                       r_cG[1] * grad_psit2_G[0]))


            # augmentation contributions to magnetic moment
            # <psi1| r x nabla |psi2> = <psi1| (r-Ra+Ra) x nabla |psi2>
            #                         = <psi1| (r-Ra) x nabla |psi2> + Ra x <psi1| nabla |psi2>
            rxnabla_a = np.zeros(3)
            # <psi1| (r-Ra) x nabla |psi2>
            for a, P_ni in self.calc.wfs.kpt_u[self.kpt_ind].P_ani.items():
                Pi_i = P_ni[kss_ip.occ_ind]
                Pp_i = P_ni[kss_ip.unocc_ind]
                rxnabla_iiv = self.calc.wfs.setups[a].rxnabla_iiv
                for c in range(3):
                    for i1, Pi in enumerate(Pi_i):
                        for i2, Pp in enumerate(Pp_i):
                            rxnabla_a[c] += Pi * Pp * rxnabla_iiv[i1, i2, c]

            self.calc.wfs.gd.comm.sum(rxnabla_a) # sum up from different procs


            # Ra x <psi1| nabla |psi2>
            Rxnabla_a = np.zeros(3)
            for a, P_ni in self.calc.wfs.kpt_u[self.kpt_ind].P_ani.items():
                Pi_i = P_ni[kss_ip.occ_ind]
                Pp_i = P_ni[kss_ip.unocc_ind]
                nabla_iiv = self.calc.wfs.setups[a].nabla_iiv
                Ra = (self.calc.atoms[a].position / ase.units.Bohr) - R0
                for i1, Pi in enumerate(Pi_i):
                    for i2, Pp in enumerate(Pp_i):
                        # (y pz - z py)i + (z px - x pz)j + (x py - y px)k
                        Rxnabla_a[0] += Pi * Pp * ( Ra[1] * nabla_iiv[i1, i2, 2] - Ra[2] * nabla_iiv[i1, i2, 1] )
                        Rxnabla_a[1] += Pi * Pp * ( Ra[2] * nabla_iiv[i1, i2, 0] - Ra[0] * nabla_iiv[i1, i2, 2] )
                        Rxnabla_a[2] += Pi * Pp * ( Ra[0] * nabla_iiv[i1, i2, 1] - Ra[1] * nabla_iiv[i1, i2, 0] )


            self.calc.wfs.gd.comm.sum(Rxnabla_a) # sum up from different procs

            #print (kss_ip.occ_ind, kss_ip.unocc_ind), kss_ip.dip_mom_r, rxnabla_g, rxnabla_a, Rxnabla_a

            # m_ip = -1/2c <i|r x p|p> = i/2c <i|r x nabla|p>
            # just imaginary part!!!
            kss_ip.magn_mom = ase.units.alpha / 2. * (rxnabla_g + rxnabla_a + Rxnabla_a)




        # Wait... to avoid io problems, and write KS_singles file
        self.lr_comms.parent_comm.barrier()
        if self.lr_comms.parent_comm.rank == 0:
            self.kss_file = open(self.basefilename+'.KS_singles','w')
            for kss_ip in self.kss_list:
                format = '%08d %08d  %18.12lf %18.12lf  '
                format += '%18.12lf %18.12lf %18.12lf '
                format += '%18.12lf %18.12lf %18.12lf\n'
                self.kss_file.write(format % (kss_ip.occ_ind, kss_ip.unocc_ind,
                                              kss_ip.energy_diff,
                                              kss_ip.pop_diff,
                                              kss_ip.dip_mom_r[0],
                                              kss_ip.dip_mom_r[1],
                                              kss_ip.dip_mom_r[2],
                                              kss_ip.magn_mom[0],
                                              kss_ip.magn_mom[1],
                                              kss_ip.magn_mom[2]))
            self.kss_file.close()
        self.lr_comms.parent_comm.barrier()

        self.kss_prop_ready = True      # avoid repeated work
        #self.timer.stop('Calculate KS properties')





###############################################################################


###############################################################################
# Small utility class
###############################################################################
# FIXME: USE EXISTING METHOD
class LRiPairDensity:
    """Pair density calculator class."""

    def __init__(self, density):
        """Initialization needs density instance"""
        self.density = density

    def initialize(self, kpt, n1, n2):
        """Set wave function indices."""
        self.n1 = n1
        self.n2 = n2
        self.spin = kpt.s
        self.P_ani = kpt.P_ani
        self.psit1_G = pick(kpt.psit_nG, n1)
        self.psit2_G = pick(kpt.psit_nG, n2)

    def get(self, nt_G, nt_g, rhot_g):
        """Get pair densities.

        nt_G
          Pair density without compensation charges on the coarse grid

        nt_g
          Pair density without compensation charges on the fine grid

        rhot_g
          Pair density with compensation charges on the fine grid
        """
        # Coarse grid product
        np.multiply(self.psit1_G.conj(), self.psit2_G, nt_G)
        # Interpolate to fine grid
        self.density.interpolator.apply(nt_G, nt_g)

        # Determine the compensation charges for each nucleus
        Q_aL = {}
        for a, P_ni in self.P_ani.items():
            assert P_ni.dtype == float
            # Generate density matrix
            P1_i = P_ni[self.n1]
            P2_i = P_ni[self.n2]
            D_ii = np.outer(P1_i.conj(), P2_i)
            # allowed to pack as used in the scalar product with
            # the symmetric array Delta_pL
            D_p  = pack(D_ii)
            #FIXME: CHECK THIS D_p  = pack(D_ii, tolerance=1e30)

            # Determine compensation charge coefficients:
            Q_aL[a] = np.dot(D_p, self.density.setups[a].Delta_pL)

        # Add compensation charges
        rhot_g[:] = nt_g[:]
        self.density.ghat.add(rhot_g, Q_aL)
        #print 'dens', self.density.finegd.integrate(rhot_g)



