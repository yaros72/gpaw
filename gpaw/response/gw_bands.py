import pickle

import numpy as np
from ase.utils import gcd
from ase.units import Ha
from scipy.interpolate import InterpolatedUnivariateSpline

from gpaw import GPAW
from gpaw.spinorbit import get_spinorbit_eigenvalues
from gpaw.kpt_descriptor import to1bz


class GWBands:
    """This class defines the GW_bands properties"""

    def __init__(self,
                 calc=None,
                 comm=None,
                 gw_file=None,
                 kpoints=None,
                 bandrange=None):

        self.calc = GPAW(calc, communicator=comm, txt=None)
        if gw_file is not None:
            self.gw_file = pickle.load(open(gw_file, 'rb'),encoding='bytes')
        self.kpoints = kpoints
        if bandrange is None:
            self.bandrange = np.arange(self.calc.get_number_of_bands())
        else:
            self.bandrange = bandrange

        self.gd = self.calc.wfs.gd.new_descriptor()
        self.kd = self.calc.wfs.kd

        self.acell_cv = self.gd.cell_cv
        self.bcell_cv = 2 * np.pi * self.gd.icell_cv
        self.vol = self.gd.volume
        self.BZvol = (2 * np.pi)**3 / self.vol

    def find_k_along_path(self, plot_BZ=True):
        """Finds the k-points along the bandpath present in the
           original calculation"""
        kd = self.kd
        acell_cv = self.acell_cv
        bcell_cv = self.bcell_cv
        kpoints = self.kpoints

        if plot_BZ:
            """Plotting the points in the Brillouin Zone"""
            kp_1bz = to1bz(kd.bzk_kc, acell_cv)

            bzk_kcv = np.dot(kd.bzk_kc, bcell_cv)
            kp_1bz_v = np.dot(kp_1bz, bcell_cv)

            import matplotlib.pyplot as plt
            plt.plot(bzk_kcv[:, 0], bzk_kcv[:, 1], 'xg')
            plt.plot(kp_1bz_v[:, 0], kp_1bz_v[:, 1], 'ob')
            for ik in range(1, len(kpoints)):
                kpoint1_v = np.dot(kpoints[ik], bcell_cv)
                kpoint2_v = np.dot(kpoints[ik - 1], bcell_cv)
                plt.plot([kpoint1_v[0], kpoint2_v[0]], [kpoint1_v[1],
                                                        kpoint2_v[1]], '--vr')

        """Finding the points along given directions"""
        print('Finding the kpoints along the path')
        N_c = kd.N_c
        wpts_xc = kpoints

        x_x = []
        k_xc = []
        k_x = []
        x = 0.
        X = []
        for nwpt in range(1, len(wpts_xc)):
            X.append(x)
            to_c = wpts_xc[nwpt]
            from_c = wpts_xc[nwpt - 1]
            vec_c = to_c - from_c
            print('From ', from_c, ' to ', to_c)
            Nv_c = (vec_c * N_c).round().astype(int)
            Nv = abs(gcd(gcd(Nv_c[0], Nv_c[1]), Nv_c[2]))
            print(Nv, ' points found')
            dv_c = vec_c / Nv
            dv_v = np.dot(dv_c, bcell_cv)
            dx = np.linalg.norm(dv_v)
            if nwpt == len(wpts_xc) - 1:
                # X.append(Nv * dx)
                Nv += 1
            for n in range(Nv):
                k_c = from_c + n * dv_c
                bzk_c = to1bz(np.array([k_c]), acell_cv)[0]
                ikpt = kd.where_is_q(bzk_c, kd.bzk_kc)
                x_x.append(x)
                k_xc.append(k_c)
                k_x.append(ikpt)
                x += dx
        X.append(x_x[-1])
        if plot_BZ is True:
            for ik in range(len(k_xc)):
                ktemp_xcv = np.dot(k_xc[ik], bcell_cv)
                plt.plot(ktemp_xcv[0], ktemp_xcv[1], 'xr', markersize=10)
            plt.show()

        return x_x, k_xc, k_x, X

    def get_dft_eigenvalues(self):
        Nk = len(self.calc.get_ibz_k_points())
        bands = np.arange(self.bandrange[0], self.bandrange[-1])
        e_kn = np.array([self.calc.get_eigenvalues(kpt=k)[bands]
                         for k in range(Nk)])
        return e_kn

    def get_vacuum_level(self, plot_pot=False):
        """Finds the vacuum level through Hartree potential"""
        vHt_g = self.calc.get_electrostatic_potential()
        vHt_z = np.mean(np.mean(vHt_g, axis=0), axis=0)

        if plot_pot:
            import matplotlib.pyplot as plt
            plt.plot(vHt_z)
            plt.show()
        return vHt_z[0]

    def get_spinorbit_corrections(self, return_spin=True, return_wfs=False,
                                  bands=None, gwqeh_file=None, dft=False,
                                  eig_file=None):
        """Gets the spinorbit corrections to the eigenvalues"""
        calc = self.calc
        bandrange = self.bandrange

        if not dft:
            try:
                e_kn = self.gw_file['qp'][0]
            except:
                e_kn = self.gw_file[b'qp'][0]
        else:
            if eig_file is not None:
                e_kn = pickle.load(open(eig_file))[0]
            else:
                e_kn = self.get_dft_eigenvalues()[:,bandrange[0]:bandrange[-1]]

        eSO_nk, s_nk, v_knm = get_spinorbit_eigenvalues(
            calc,
            return_spin=return_spin, return_wfs=return_wfs,
            bands=range(bandrange[0], bandrange[-1]+1),
            gw_kn=e_kn)

        e_kn = eSO_nk.T
        return e_kn, v_knm

    def get_gw_bands(self, nk_Int=50, interpolate=False, SO=False,
                     gwqeh_file=None, dft=False, eig_file=None, vac=False):
        """Getting Eigenvalues along the path"""
        kd = self.kd
        if SO:
            e_kn, v_knm = self.get_spinorbit_corrections(return_wfs=True,
                                                         dft=dft,
                                                         eig_file=eig_file)
            if gwqeh_file is not None:
                gwqeh_file = pickle.load(open(gwqeh_file))
                eqeh_noSO_kn = gwqeh_file['qp_sin'][0] * Ha
                eqeh_kn = np.zeros_like(e_kn)
                eqeh_kn[:, ::2] = eqeh_noSO_kn
                eqeh_kn[:, 1::2] = eqeh_noSO_kn

                e_kn += eqeh_kn

        elif gwqeh_file is not None:
            gwqeh_file = pickle.load(open(gwqeh_file))
            e_kn = gwqeh_file['Qp_sin'][0] * Ha
        elif eig_file is not None:
            e_kn = pickle.load(open(eig_file))[0]
        else:
            if not dft:
                try:
                    e_kn = self.gw_file['qp'][0]
                except:
                    e_kn = self.gw_file[b'qp'][0]
            else:
                e_kn = self.get_dft_eigenvalues()
        e_kn = np.sort(e_kn, axis=1)

        bandrange = self.bandrange
        ef = self.calc.get_fermi_level()
        if vac:
            evac = self.get_vacuum_level()
        else:
            evac = 0.0
        x_x, k_xc, k_x, X = self.find_k_along_path(plot_BZ=False)

        k_ibz_x = np.zeros_like(k_x)
        eGW_kn = np.zeros((len(k_x), e_kn.shape[1]))
        for n in range(e_kn.shape[1]):
            for ik in range(len(k_x)):
                ibzkpt = kd.bz2ibz_k[k_x[ik]]
                k_ibz_x[ik] = ibzkpt
                eGW_kn[ik, n] = e_kn[ibzkpt, n]

        N_occ = (eGW_kn[0] < ef).sum()
        print(N_occ, bandrange[0])
        # N_occ = int(self.calc.get_number_of_electrons()/2)
        print(' ')
        if SO:
            print('The number of Occupied bands is:', N_occ + 2.*bandrange[0])
        else:
            print('The number of Occupied bands is:', N_occ + bandrange[0])
        gap = (eGW_kn[:, N_occ].min() - eGW_kn[:, N_occ - 1].max())
        print('The bandgap is: %f' % gap)

        vbm = eGW_kn[:, N_occ - 1].max() - evac
        cbm = eGW_kn[:, N_occ].min() - evac

        if interpolate:
            xfit_k = np.linspace(x_x[0], x_x[-1], nk_Int)
            xfit_k = np.append(xfit_k, x_x)
            xfit_k = np.sort(xfit_k)
            nk_Int = len(xfit_k)
            efit_kn = np.zeros((nk_Int, eGW_kn.shape[1]))
            for n in range(eGW_kn.shape[1]):
                fit_e = InterpolatedUnivariateSpline(x_x, eGW_kn[:, n])
                efit_kn[:, n] = fit_e(xfit_k)

            results = {'x_k': xfit_k,
                       'X': X,
                       'e_kn': efit_kn - evac,
                       'ef': ef - evac,
                       'gap': gap,
                       'vbm': vbm,
                       'cbm': cbm}

            return results
        else:
            results = {'x_k': x_x,
                       'X': X,
                       'k_ibz_x': k_ibz_x,
                       'e_kn': eGW_kn - evac,
                       'ef': ef - evac,
                       'gap': gap,
                       'vbm': vbm,
                       'cbm': cbm}
            return results

