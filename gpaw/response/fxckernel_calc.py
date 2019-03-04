from ase.units import Ha
import os
import gpaw.mpi as mpi
import numpy as np
from gpaw.xc.fxc import KernelWave
from ase.io.aff import affopen

def calculate_kernel(self, nG, ns, iq, cut_G=None):
    self.unit_cells = self.calc.wfs.kd.N_c
    self.tag = self.calc.atoms.get_chemical_formula(mode='hill')

    if self.av_scheme is not None:
        self.tag += '_' + self.av_scheme + '_nspins' + str(self.nspins)

    kd = self.calc.wfs.kd
    self.bzq_qc = kd.get_bz_q_points(first=True)
    U_scc = kd.symmetry.op_scc
    self.ibzq_qc = kd.get_ibz_q_points(self.bzq_qc, U_scc)[0]

    ecut = self.ecut * Ha
    if isinstance(ecut, (float, int)):
        self.ecut_max = ecut
    else:
        self.ecut_max = max(ecut)

    q_empty = None

    if not os.path.isfile('fhxc_%s_%s_%s_%s.ulm'
                          % (self.tag, self.xc,
                             self.ecut_max, iq)):
        q_empty = iq
        
    if self.xc not in ('RPA'):
        if q_empty is not None:
            self.l_l = np.array([1.0])

            if self.linear_kernel:
                kernel = KernelWave(self.calc,
                                    self.xc,
                                    self.ibzq_qc,
                                    self.fd,
                                    None,
                                    q_empty,
                                    None,
                                    self.Eg,
                                    self.ecut_max,
                                    self.tag,
                                    self.timer)

            elif not self.dyn_kernel:
                kernel = KernelWave(self.calc,
                                    self.xc,
                                    self.ibzq_qc,
                                    self.fd,
                                    self.l_l,
                                    q_empty,
                                    None,
                                    self.Eg,
                                    self.ecut_max,
                                    self.tag,
                                    self.timer)

            else:
                kernel = KernelWave(self.calc,
                                    self.xc,
                                    self.ibzq_qc,
                                    self.fd,
                                    self.l_l,
                                    q_empty,
                                    self.omega_w,
                                    self.Eg,
                                    self.ecut_max,
                                    self.tag,
                                    self.timer)

            kernel.calculate_fhxc()
            del kernel

        mpi.world.barrier()

        if self.spin_kernel:
            r = affopen('fhxc_%s_%s_%s_%s.ulm' %
                        (self.tag, self.xc, self.ecut_max, iq))
            fv = r.fhxc_sGsG
        
            if cut_G is not None:
                cut_sG = np.tile(cut_G, ns)
                cut_sG[len(cut_G):] += len(fv) // ns
                fv = fv.take(cut_sG, 0).take(cut_sG, 1)

        else:
            if self.xc == 'RPA':
                fv = np.eye(nG)
            elif self.xc == 'range_RPA':
                raise NotImplementedError
#                    fv = np.exp(-0.25 * (G_G * self.range_rc) ** 2.0)

            elif self.linear_kernel:
                r = affopen('fhxc_%s_%s_%s_%s.ulm' %
                            (self.tag, self.xc, self.ecut_max, iq))
                fv = r.fhxc_sGsG
                
                if cut_G is not None:
                    fv = fv.take(cut_G, 0).take(cut_G, 1)
                        
            elif not self.dyn_kernel:
                # static kernel which does not scale with lambda
                
                r = affopen('fhxc_%s_%s_%s_%s.ulm' %
                            (self.tag, self.xc, self.ecut_max, iq))
                fv = r.fhxc_lGG
            
                if cut_G is not None:
                    fv = fv.take(cut_G, 1).take(cut_G, 2)

            else:  # dynamical kernel
                r = affopen('fhxc_%s_%s_%s_%s.ulm' %
                            (self.tag, self.xc, self.ecut_max, iq))
                fv = r.fhxc_lwGG
                        
                if cut_G is not None:
                    fv = fv.take(cut_G, 2).take(cut_G, 3)
    else:
        fv = np.eye(nG)

    return fv

