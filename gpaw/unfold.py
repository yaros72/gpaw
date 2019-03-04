import numpy as np
import pickle

from ase.units import Hartree

from gpaw import GPAW
from gpaw.kpt_descriptor import to1bz
from gpaw.spinorbit import get_spinorbit_eigenvalues
from gpaw.wavefunctions.pw import PWDescriptor
import gpaw.mpi as mpi


class Unfold:
    """This Class is used to Unfold the Bands of a supercell (SC) calculations
    into a the primitive cell (PC). As a convention (when possible) capital
    letters variables are related to the SC while lowercase ones to the
    PC """

    def __init__(self,
                 name=None,
                 calc=None,
                 M=None,
                 spinorbit=None):

        self.name = name
        self.calc = GPAW(calc, txt=None, communicator=mpi.serial_comm)
        self.M = np.array(M, dtype=float)
        self.spinorbit = spinorbit

        self.gd = self.calc.wfs.gd.new_descriptor()

        self.kd = self.calc.wfs.kd
        if self.calc.wfs.mode == 'pw':
            self.pd = self.calc.wfs.pd
        else:
            self.pd = PWDescriptor(ecut=None, gd=self.gd, kd=self.kd,
                                   dtype=complex)

        self.acell_cv = self.gd.cell_cv
        self.bcell_cv = 2 * np.pi * self.gd.icell_cv
        self.vol = self.gd.volume
        self.BZvol = (2 * np.pi)**3 / self.vol

        self.nb = self.calc.get_number_of_bands()

        self.v_Knm = None
        if spinorbit:
            if mpi.world.rank == 0:
                print('Calculating spinorbit Corrections')
            self.nb = 2 * self.calc.get_number_of_bands()
            self.e_mK, self.v_Knm = get_spinorbit_eigenvalues(self.calc,
                                                              return_wfs=True)
            if mpi.world.rank == 0:
                print('Done with the spinorbit Corrections')

    def get_K_index(self, K):
        """Find the index of a given K."""

        K = np.array([K])
        bzKG = to1bz(K, self.acell_cv)[0]
        iK = self.kd.where_is_q(bzKG, self.kd.bzk_kc)
        return iK

    def get_g(self, iK):
        """Not all the G vectors are relevant for the bands unfolding,
        but only the ones that match the PC reciprocal vectors.
        This function finds the relevant ones."""

        G_Gv_temp = self.pd.get_reciprocal_vectors(q=iK, add_q=False)
        G_Gc_temp = np.dot(G_Gv_temp, np.linalg.inv(self.bcell_cv))

        iG_list = []
        g_list = []
        for iG, G in enumerate(G_Gc_temp):
            a = np.dot(G, np.linalg.inv(self.M).T)
            check = np.abs(a) % 1 < 1e-5
            check2 = np.abs((np.abs(a[np.where(~check)]) % 1) - 1) < 1e-5
            if all(check) or all(check2):
                iG_list.append(iG)
                g_list.append(G)

        return np.array(iG_list), np.array(g_list)

    def get_G_index(self, iK, G, G_list):
        """Find the index of a given G."""

        G_list -= G
        sumG = np.sum(abs(G_list), axis=1)
        iG = np.where(sumG < 1e-5)[0]
        return iG

    def get_eigenvalues(self, iK):
        """Get the list of eigenvalues for a given iK."""

        if not self.spinorbit:
            e_m = self.calc.get_eigenvalues(kpt=iK, spin=0) / Hartree
        else:
            e_m = self.e_mK[:, iK] / Hartree
        return np.array(e_m)

    def get_pw_wavefunctions_k(self, iK):
        """Get the list of Fourier coefficients of the WaveFunction for a
        given iK. For spinors the number of bands is doubled and a spin
        dimension is added."""

        psi_mgrid = get_rs_wavefunctions_k(self.calc, iK, self.spinorbit,
                                           self.v_Knm)
        if not self.spinorbit:
            psi_list_mG = []
            for i in range(len(psi_mgrid)):
                psi_list_mG.append(self.pd.fft(psi_mgrid[i], iK))

            psi_mG = np.array(psi_list_mG)
            return psi_mG
        else:
            u0_list_mG = []
            u1_list_mG = []
            for i in range(psi_mgrid.shape[0]):
                u0_list_mG.append(self.pd.fft(psi_mgrid[i, 0], iK))
                u1_list_mG.append(self.pd.fft(psi_mgrid[i, 1], iK))

            u0_mG = np.array(u0_list_mG)
            u1_mG = np.array(u1_list_mG)

            u_mG = np.zeros((len(u0_mG),
                             2,
                             u0_mG.shape[1]), complex)

            u_mG[:, 0] = u0_mG
            u_mG[:, 1] = u1_mG
            return u_mG

    def get_spectral_weights_k(self, k_t):
        """Returns the spectral weights for a given k in the PC:

            P_mK(k_t) = \sum_n |<Km|k_t n>|**2

        which can be shown to be equivalent to:

            P_mK(k_t) = \sum_g |C_Km(g+k_t-K)|**2
        """

        K_c, G_t = find_K_from_k(k_t, self.M)
        iK = self.get_K_index(K_c)
        iG_list, g_list = self.get_g(iK)
        gG_t_list = g_list + G_t

        G_Gv = self.pd.get_reciprocal_vectors(q=iK, add_q=False)
        G_Gc = np.dot(G_Gv, np.linalg.inv(self.bcell_cv))

        igG_t_list = []
        for g in gG_t_list:
            igG_t_list.append(self.get_G_index(iK, g.copy(), G_Gc.copy()))

        C_mG = self.get_pw_wavefunctions_k(iK)
        P_m = []
        if not self.spinorbit:
            for m in range(self.nb):
                P = 0.
                norm = np.sum(np.linalg.norm(C_mG[m, :])**2)
                for iG in igG_t_list:
                    P += np.linalg.norm(C_mG[m, iG])**2
                P_m.append(P / norm)
        else:
            for m in range(self.nb):
                P = 0.
                norm = np.sum(np.linalg.norm(C_mG[m, 0, :])**2 +
                              np.linalg.norm(C_mG[m, 1, :])**2)
                for iG in igG_t_list:
                    P += (np.linalg.norm(C_mG[m, 0, iG])**2 +
                          np.linalg.norm(C_mG[m, 1, iG])**2)
                P_m.append(P / norm)

        return np.array(P_m)

    def get_spectral_weights(self, kpoints, filename=None):
        """Collect the spectral weights for the k points in the kpoints list.

        This function is parallelized over k's."""

        Nk = len(kpoints)
        Nb = self.nb

        world = mpi.world
        if filename is None:
            try:
                e_mK, P_mK = pickle.load(open('weights_' + self.name +
                                              '.pckl', 'rb'))
            except IOError:
                e_Km = []
                P_Km = []
                if world.rank == 0:
                    print('Getting EigenValues and Weights')

                e_Km = np.zeros((Nk, Nb))
                P_Km = np.zeros((Nk, Nb))
                myk = range(0, Nk)[world.rank::world.size]
                for ik in myk:
                    k = kpoints[ik]
                    print('kpoint: %s' % k)
                    K_c, G_c = find_K_from_k(k, self.M)
                    iK = self.get_K_index(K_c)
                    e_Km[ik] = self.get_eigenvalues(iK)
                    P_Km[ik] = self.get_spectral_weights_k(k)

                world.barrier()
                world.sum(e_Km)
                world.sum(P_Km)

                e_mK = np.array(e_Km).T
                P_mK = np.array(P_Km).T
                if world.rank == 0:
                    pickle.dump((e_mK, P_mK),
                                open('weights_' + self.name + '.pckl', 'wb'))
        else:
            e_mK, P_mK = pickle.load(open(filename, 'rb'))

        return e_mK, P_mK

    def spectral_function(self, kpts, x, X, points_name, width=0.002,
                          npts=10000, filename=None):
        """Returns the spectral function for all the ks in kpoints:

                                              eta / pi

            A_k(e) = \sum_m  P_mK(k) x  ---------------------

                                        (e - e_mk)**2 + eta**2


        at each k-points defined on npts energy points in the range
        [emin, emax]. The width keyword is FWHM = 2 * eta."""

        Nk = len(kpts)
        A_ke = np.zeros((Nk, npts), float)

        world = mpi.world
        e_mK, P_mK = self.get_spectral_weights(kpts, filename)
        if world.rank == 0:
            print('Calculating the Spectral Function')
        emin = np.min(e_mK) - 5 * width
        emax = np.max(e_mK) + 5 * width
        e = np.linspace(emin, emax, npts)

        for ik in range(Nk):
            for ie in range(len(e_mK[:, ik])):
                e0 = e_mK[ie, ik]
                D = (width / 2 / np.pi) / ((e - e0)**2 + (width / 2)**2)
                A_ke[ik] += P_mK[ie, ik] * D
        if world.rank == 0:
            pickle.dump((e * Hartree, A_ke, x, X, points_name),
                        open('sf_' + self.name + '.pckl', 'wb'))
            print('Spectral Function calculation completed!')
        return


def find_K_from_k(k, M):
    """Gets a k vector in scaled coordinates and returns a K vector and the
    unfolding G in scaled Coordinates."""

    KG = np.dot(M, k)
    G = np.zeros(3, dtype=int)

    for i in range(3):
        if KG[i] > 0.5000001:
            G[i] = int(np.round(KG[i]))
            KG[i] -= np.round(KG[i])
        elif KG[i] < -0.4999999:
            G[i] = int(np.round(KG[i]))
            KG[i] += abs(np.round(KG[i]))

    return KG, G


def get_rs_wavefunctions_k(calc, iK, spinorbit=False, v_Knm=None):
    """Get the list of WaveFunction for a given iK. For spinors the number of
    bands is doubled and a spin dimension is added."""

    N_c = calc.wfs.gd.N_c
    k_c = calc.wfs.kd.ibzk_kc[iK]
    Nb = calc.wfs.bd.nbands
    Ns = calc.wfs.nspins
    eikr_R = np.exp(-2j * np.pi * np.dot(np.indices(N_c).T,
                                         k_c / N_c).T)

    if calc.wfs.mode == 'lcao' and not calc.wfs.positions_set:
        calc.initialize_positions()

    if not spinorbit:
        psit_mgrid = np.array([calc.wfs.get_wave_function_array(m, iK, 0) *
                               eikr_R for m in range(Nb)])
        return psit_mgrid
    else:
        v_nm = v_Knm[iK].copy()
        v0_mn = v_nm[::2].T
        v1_mn = v_nm[1::2].T

        u0_ngrid = np.array(
            [calc.wfs.get_wave_function_array(n, iK, 0) * eikr_R
             for n in range(Nb)])
        u1_ngrid = np.array(
            [calc.wfs.get_wave_function_array(n, iK, (Ns - 1)) * eikr_R
             for n in range(Nb)])

        u0_mG = np.swapaxes(np.dot(v0_mn, np.swapaxes(u0_ngrid, 0, 2)), 1, 2)
        u1_mG = np.swapaxes(np.dot(v1_mn, np.swapaxes(u1_ngrid, 0, 2)), 1, 2)
        ut_mgrid = np.zeros((len(u0_mG),
                             2,
                             len(u0_mG[0]),
                             len(u0_mG[0, 0]),
                             len(u0_mG[0, 0, 0])), complex)
        ut_mgrid[:, 0] = u0_mG
        ut_mgrid[:, 1] = u1_mG
        return ut_mgrid


def plot_spectral_function(filename, color='blue', eref=None,
                           emin=None, emax=None):
    """Function to plot spectral function corresponding to the bandstructure
    along the kpoints path."""

    try:
        e, A_ke, x, X, points_name = pickle.load(open(filename + '.pckl',
                                                      'rb'))
    except IOError:
        print('You Need to Calculate the SF first!')
        raise SystemExit()

    import matplotlib.pyplot as plt
    print('Plotting Spectral Function')

    if eref is not None:
        e -= eref
    if emin is None:
        emin = e.min()
    if emax is None:
        emax = e.max()

    A_ke /= np.max(A_ke)
    A_ek = A_ke.T
    A_ekc = np.reshape(A_ek, (A_ek.shape[0], A_ek.shape[1]))

    mycmap = make_colormap(color)

    plt.figure()

    plt.plot([0, x[-1]], 2 * [0.0], '--', c='0.5')
    plt.imshow(A_ekc + 0.23,
               cmap=mycmap,
               aspect='auto',
               origin='lower',
               vmin=0.,
               vmax=1,
               extent=[0, x[-1], e.min(), e.max()])

    for k in X[1:-1]:
        plt.plot([k, k], [emin, emax], lw=0.5, c='0.5')
    plt.xticks(X, points_name, size=20)
    plt.yticks(size=20)
    plt.ylabel('E(eV)', size=20)
    plt.axis([0, x[-1], emin, emax])
    plt.savefig(filename + '.png')
    plt.show()


def plot_band_structure(e_mK, P_mK, x, X, points_name,
                        weights_mK=None, color='red', fit=True, nfit=200):
    """Function to plot the bandstructure using the P_mK weights directly.
    each point is represented with a filled circle, whose size and color
    vary with as a function of P_mK."""

    import matplotlib.pyplot as plt
    print('Plotting Bands Structure')
    emin = e_mK.min()
    emax = e_mK.max()

    new_cmap = make_colormap(color)

    if weights_mK is None:
        weights_mK = P_mK.copy()
    else:
        weights_mK *= P_mK.copy()

    plt.figure()
    plt.plot([0, x[-1]], 2 * [0.0], '--', c='0.5')
    plt.scatter(np.tile(x, len(e_mK)), e_mK.reshape(-1),
                c=P_mK.reshape(-1),
                cmap=new_cmap,
                vmin=0.,
                vmax=1.,
                s=20. * weights_mK.reshape(-1),
                marker='o',
                edgecolor='none')

    for k in X[1:-1]:
        plt.plot([k, k], [emin, emax], lw=0.5, c='0.5')
    plt.xticks(X, points_name, size=20)
    plt.yticks(size=20)
    plt.ylabel('E(eV)', size=20)
    plt.axis([0, x[-1], emin, emax])
    plt.show()


def make_colormap(main_color):
    """Custom colormaps used in plot_spectral function and
    plot_band_structure."""

    from matplotlib.colors import LinearSegmentedColormap
    if main_color == 'blue':
        cdict = {'red': ((0.0, 1.0, 1.0),
                         (0.25, 1.0, 1.0),
                         (0.5, 0.0, 0.0),
                         (1.0, 0.0, 0.0)),

                 'green': ((0.0, 1.0, 1.0),
                           (0.25, 1.0, 1.0),
                           (0.5, 0.0, 0.0),
                           (1.0, 0.0, 0.0)),

                 'blue': ((0.0, 1.0, 1.0),
                          (0.25, 1.0, 1.0),
                          (0.5, 1.0, 1.0),
                          (1.0, 0.75, 0.75))}

    elif main_color == 'red':
        cdict = {'red': ((0.0, 1.0, 1.0),
                         (0.25, 1.0, 1.0),
                         (0.5, 1.0, 1.0),
                         (1.0, 0.75, 0.75)),

                 'green': ((0.0, 1.0, 1.0),
                           (0.25, 1.0, 1.0),
                           (0.5, 0.0, 0.0),
                           (1.0, 0.0, 0.0)),

                 'blue': ((0.0, 1.0, 1.0),
                          (0.25, 1.0, 1.0),
                          (0.5, 0.0, 0.0),
                          (1.0, 0.0, 0.0))}

    elif main_color == 'green':
        cdict = {'red': ((0.0, 1.0, 1.0),
                         (0.25, 1.0, 1.0),
                         (0.5, 0.0, 0.0),
                         (1.0, 0.0, 0.0)),

                 'green': ((0.0, 1.0, 1.0),
                           (0.25, 1.0, 1.0),
                           (0.5, 1.0, 1.0),
                           (1.0, 0.75, 0.75)),

                 'blue': ((0.0, 1.0, 1.0),
                          (0.25, 1.0, 1.0),
                          (0.5, 0.0, 0.0),
                          (1.0, 0.0, 0.0)),

                 'alpha': ((0.0, 0.0, 0.0),
                           (0.25, 0.1, 0.1),
                           (0.5, 1.0, 1.0),
                           (1.0, 1.0, 1.0))}

    cmap = LinearSegmentedColormap('mymap', cdict)
    return cmap


def get_vacuum_level(calc, plot_pot=False):
    """Get the vacuum energy level from a given calculator."""

    calc.restore_state()
    if calc.wfs.mode == 'pw':
        vHt_g = calc.hamiltonian.pd3.ifft(calc.hamiltonian.vHt_q) * Hartree
    else:
        vHt_g = calc.hamiltonian.vHt_g * Hartree
    vHt_z = np.mean(np.mean(vHt_g, axis=0), axis=0)

    if plot_pot:
        import matplotlib.pyplot as plt
        plt.plot(vHt_z)
        plt.show()
    return vHt_z[0]
