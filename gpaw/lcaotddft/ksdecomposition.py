# -*- coding: utf-8 -*-
import numpy as np

from ase.units import Hartree, Bohr

from gpaw.io import Reader
from gpaw.io import Writer
from gpaw.external import ConstantElectricField
from gpaw.lcaotddft.hamiltonian import KickHamiltonian
from gpaw.lcaotddft.utilities import read_uMM
from gpaw.lcaotddft.utilities import write_uMM
from gpaw.utilities.tools import tri2full


class KohnShamDecomposition(object):
    version = 1
    ulmtag = 'KSD'
    readwrite_attrs = ['fermilevel', 'only_ia', 'w_p', 'f_p', 'ia_p',
                       'P_p', 'dm_vp']

    def __init__(self, paw=None, filename=None):
        self.filename = filename
        self.has_initialized = False
        self.world = paw.world
        self.log = paw.log
        self.wfs = paw.wfs
        self.density = paw.density

        if self.wfs.bd.comm.size > 1:
            raise RuntimeError('Band parallelization is not supported')
        if len(self.wfs.kpt_u) > 1:
            raise RuntimeError('K-points are not supported')

        if filename is not None:
            self.read(filename)
            return

    def initialize(self, paw, min_occdiff=1e-3, only_ia=True):
        if self.has_initialized:
            return
        paw.initialize_positions()
        # paw.set_positions()

        if self.wfs.gd.pbc_c.any():
            self.C0_dtype = complex
        else:
            self.C0_dtype = float

        # Take quantities
        self.fermilevel = paw.occupations.get_fermi_level()
        self.S_uMM = []
        self.C0_unM = []
        self.eig_un = []
        self.occ_un = []
        for kpt in self.wfs.kpt_u:
            S_MM = kpt.S_MM
            assert np.max(np.absolute(S_MM.imag)) == 0.0
            S_MM = S_MM.real
            self.S_uMM.append(S_MM)

            C_nM = kpt.C_nM
            if self.C0_dtype == float:
                assert np.max(np.absolute(C_nM.imag)) == 0.0
                C_nM = C_nM.real
            self.C0_unM.append(C_nM)

            self.eig_un.append(kpt.eps_n)
            self.occ_un.append(kpt.f_n)

        # TODO: do the rest of the function with K-points

        # Construct p = (i, a) pairs
        u = 0
        Nn = self.wfs.bd.nbands
        eig_n = self.eig_un[u]
        occ_n = self.occ_un[u]

        self.only_ia = only_ia
        f_p = []
        w_p = []
        i_p = []
        a_p = []
        ia_p = []
        i0 = 0
        for i in range(i0, Nn):
            if only_ia:
                a0 = i + 1
            else:
                a0 = 0
            for a in range(a0, Nn):
                f = occ_n[i] - occ_n[a]
                if only_ia and f < min_occdiff:
                    continue
                w = eig_n[a] - eig_n[i]
                f_p.append(f)
                w_p.append(w)
                i_p.append(i)
                a_p.append(a)
                ia_p.append((i, a))
        f_p = np.array(f_p)
        w_p = np.array(w_p)
        i_p = np.array(i_p, dtype=int)
        a_p = np.array(a_p, dtype=int)
        ia_p = np.array(ia_p, dtype=int)

        # Sort according to energy difference
        p_s = np.argsort(w_p)
        f_p = f_p[p_s]
        w_p = w_p[p_s]
        i_p = i_p[p_s]
        a_p = a_p[p_s]
        ia_p = ia_p[p_s]

        Np = len(f_p)
        P_p = []
        for p in range(Np):
            P = np.ravel_multi_index(ia_p[p], (Nn, Nn))
            P_p.append(P)
        P_p = np.array(P_p)

        dm_vMM = []
        for v in range(3):
            direction = np.zeros(3, dtype=float)
            direction[v] = 1.0
            magnitude = 1.0
            cef = ConstantElectricField(magnitude * Hartree / Bohr, direction)
            kick_hamiltonian = KickHamiltonian(paw, cef)
            dm_MM = self.wfs.eigensolver.calculate_hamiltonian_matrix(
                kick_hamiltonian, paw.wfs, self.wfs.kpt_u[u],
                add_kinetic=False, root=-1)
            tri2full(dm_MM)  # TODO: do not use this
            dm_vMM.append(dm_MM)

        C0_nM = self.C0_unM[u]
        dm_vnn = []
        for v in range(3):
            dm_vnn.append(np.dot(C0_nM.conj(), np.dot(dm_vMM[v], C0_nM.T)))
        dm_vnn = np.array(dm_vnn)
        dm_vP = dm_vnn.reshape(3, -1)

        dm_vp = dm_vP[:, P_p]

        self.w_p = w_p
        self.f_p = f_p
        self.ia_p = ia_p
        self.P_p = P_p
        self.dm_vp = dm_vp

        self.has_initialized = True

    def write(self, filename):
        self.log('%s: Writing to %s' % (self.__class__.__name__, filename))
        writer = Writer(filename, self.world, mode='w',
                        tag=self.__class__.ulmtag)
        writer.write(version=self.__class__.version)

        wfs = self.wfs
        writer.write(ha=Hartree)
        write_uMM(wfs, writer, 'S_uMM', self.S_uMM)
        wfs.write_wave_functions(writer)
        wfs.write_eigenvalues(writer)
        wfs.write_occupations(writer)
        # write_unM(wfs, writer, 'C0_unM', self.C0_unM)
        # write_un(wfs, writer, 'eig_un', self.eig_un)
        # write_un(wfs, writer, 'occ_un', self.occ_un)

        for arg in self.readwrite_attrs:
            writer.write(arg, getattr(self, arg))

        writer.close()

    def read(self, filename):
        reader = Reader(filename)
        tag = reader.get_tag()
        if tag != self.__class__.ulmtag:
            raise RuntimeError('Unknown tag %s' % tag)
        version = reader.version
        if version != self.__class__.version:
            raise RuntimeError('Unknown version %s' % version)

        wfs = self.wfs
        self.S_uMM = read_uMM(wfs, reader, 'S_uMM')
        wfs.read_wave_functions(reader)
        wfs.read_eigenvalues(reader)
        wfs.read_occupations(reader)

        self.C0_unM = []
        self.eig_un = []
        self.occ_un = []
        for kpt in self.wfs.kpt_u:
            C_nM = kpt.C_nM
            self.C0_unM.append(C_nM)
            self.eig_un.append(kpt.eps_n)
            self.occ_un.append(kpt.f_n)

        for arg in self.readwrite_attrs:
            setattr(self, arg, getattr(reader, arg))

        reader.close()
        self.has_initialized = True

    def transform(self, rho_uMM):
        assert len(rho_uMM) == 1, 'K-points not implemented'
        u = 0
        C0_nM = self.C0_unM[u]
        S_MM = self.S_uMM[u]
        rho_MM = rho_uMM[u]
        # KS decomposition
        C0S_nM = np.dot(C0_nM, S_MM)
        rho_nn = np.dot(np.dot(C0S_nM, rho_MM), C0S_nM.T.conjugate())
        rho_P = rho_nn.reshape(-1)

        # Remove de-excitation terms
        rho_p = rho_P[self.P_p]
        if self.only_ia:
            rho_p *= 2

        rho_up = [rho_p]
        return rho_up

    def ialims(self):
        i_p = self.ia_p[:, 0]
        a_p = self.ia_p[:, 1]
        imin = np.min(i_p)
        imax = np.max(i_p)
        amin = np.min(a_p)
        amax = np.max(a_p)
        return imin, imax, amin, amax

    def M_p_to_M_ia(self, M_p):
        return self.M_ia_from_M_p(M_p)

    def M_ia_from_M_p(self, M_p):
        imin, imax, amin, amax = self.ialims()
        M_ia = np.zeros((imax - imin + 1, amax - amin + 1), dtype=M_p.dtype)
        for M, (i, a) in zip(M_p, self.ia_p):
            M_ia[i - imin, a - amin] = M
        return M_ia

    def plot_matrix(self, M_p):
        import matplotlib.pyplot as plt
        M_ia = self.M_ia_from_M_p(M_p)
        plt.imshow(M_ia, interpolation='none')
        plt.xlabel('a')
        plt.ylabel('i')

    def get_dipole_moment_contributions(self, rho_up):
        assert len(rho_up) == 1, 'K-points not implemented'
        u = 0
        rho_p = rho_up[u]
        dmrho_vp = - self.dm_vp * rho_p
        return dmrho_vp

    def get_dipole_moment(self, rho_up):
        assert len(rho_up) == 1, 'K-points not implemented'
        u = 0
        rho_p = rho_up[u]
        dm_v = - np.dot(self.dm_vp, rho_p)
        return dm_v

    def get_density(self, rho_up, density='comp'):
        from gpaw.lcaotddft.densitymatrix import get_density

        density_type = density
        assert len(rho_up) == 1, 'K-points not implemented'
        u = 0
        rho_p = rho_up[u]
        C0_nM = self.C0_unM[u]

        rho_ia = self.M_ia_from_M_p(rho_p)
        imin, imax, amin, amax = self.ialims()
        C0_iM = C0_nM[imin:(imax + 1)]
        C0_aM = C0_nM[amin:(amax + 1)]

        rho_MM = np.dot(C0_iM.T, np.dot(rho_ia, C0_aM.conj()))
        rho_MM = 0.5 * (rho_MM + rho_MM.T)

        return get_density(rho_MM, self.wfs, self.density, density_type, u)

    def get_contributions_table(self, weight_p, minweight=0.01,
                                zero_fermilevel=True):
        assert weight_p.dtype == float
        u = 0  # TODO

        absweight_p = np.absolute(weight_p)
        tot_weight = weight_p.sum()
        rest_weight = tot_weight
        eig_n = self.eig_un[u].copy()
        if zero_fermilevel:
            eig_n -= self.fermilevel

        txt = ''
        txt += ('# %6s %3s(%8s)    %3s(%8s)  %12s %14s\n' %
                ('p', 'i', 'eV', 'a', 'eV', 'Ediff (eV)', 'weight'))
        p_s = np.argsort(absweight_p)[::-1]
        for s, p in enumerate(p_s):
            i, a = self.ia_p[p]
            if absweight_p[p] < minweight:
                break
            txt += ('  %6s %3d(%8.3f) -> %3d(%8.3f): %12.4f %+14.4f\n' %
                    (p, i, eig_n[i] * Hartree, a, eig_n[a] * Hartree,
                     self.w_p[p] * Hartree, weight_p[p]))
            rest_weight -= weight_p[p]
        txt += ('  %37s: %12s %+14.4f\n' %
                ('rest', '', rest_weight))
        txt += ('  %37s: %12s %+14.4f\n' %
                ('total', '', tot_weight))
        return txt

    def plot_TCM(self, weight_p,
                 occ_energy_min, occ_energy_max,
                 unocc_energy_min, unocc_energy_max,
                 delta_energy, sigma, zero_fermilevel=True,
                 spectrum=False,
                 vmax='80%'):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # Calculate TCM
        args = (weight_p,
                occ_energy_min, occ_energy_max,
                unocc_energy_min, unocc_energy_max,
                delta_energy, sigma, zero_fermilevel)
        r = self.get_TCM(*args)
        energy_o, energy_u, dos_o, dos_u, tcm_ou, fermilevel = r

        # Start plotting
        plt.figure(figsize=(8, 8))
        linecolor = 'k'

        # Generate axis
        def get_gs(**kwargs):
            width = 0.84
            bottom = 0.12
            left = 0.12
            return GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3],
                            bottom=bottom, top=bottom + width,
                            left=left, right=left + width,
                            **kwargs)
        gs = get_gs(hspace=0.05, wspace=0.05)
        ax_occ_dos = plt.subplot(gs[0])
        ax_unocc_dos = plt.subplot(gs[3])
        ax_tcm = plt.subplot(gs[2])
        if spectrum:
            ax_spec = plt.subplot(get_gs(hspace=0.8, wspace=0.8)[1])
        else:
            ax_spec = None

        # Plot TCM
        ax = ax_tcm
        plt.sca(ax)
        if isinstance(vmax, str):
            assert vmax[-1] == '%'
            tcmmax = max(np.max(tcm_ou), -np.min(tcm_ou))
            vmax = tcmmax * float(vmax[:-1]) / 100.0
        vmin = -vmax
        cmap = 'seismic'
        plt.pcolormesh(energy_o, energy_u, tcm_ou.T,
                       cmap=cmap, rasterized=True, vmin=vmin, vmax=vmax)
        plt.axhline(fermilevel, c=linecolor)
        plt.axvline(fermilevel, c=linecolor)

        ax.tick_params(axis='both', which='major', pad=2)
        plt.xlabel(r'Occ. energy $\varepsilon_{o}$ (eV)', labelpad=0)
        plt.ylabel(r'Unocc. energy $\varepsilon_{u}$ (eV)', labelpad=0)
        plt.xlim(occ_energy_min, occ_energy_max)
        plt.ylim(unocc_energy_min, unocc_energy_max)

        # Plot DOSes
        def plot_DOS(ax, energy_e, dos_e,
                     energy_min, energy_max,
                     dos_min, dos_max,
                     flip=False):
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            if flip:
                set_label = ax.set_xlabel
                fill_between = ax.fill_betweenx
                set_energy_lim = ax.set_ylim
                set_dos_lim = ax.set_xlim

                def plot(x, y, *args, **kwargs):
                    return ax.plot(y, x, *args, **kwargs)
            else:
                set_label = ax.set_ylabel
                fill_between = ax.fill_between
                set_energy_lim = ax.set_xlim
                set_dos_lim = ax.set_ylim

                def plot(x, y, *args, **kwargs):
                    return ax.plot(x, y, *args, **kwargs)
            fill_between(energy_e, 0, dos_e, color='0.8')
            plot(energy_e, dos_e, 'k')
            set_label('DOS', labelpad=0)
            set_energy_lim(energy_min, energy_max)
            set_dos_lim(dos_min, dos_max)

        dos_min = 0.0
        dos_max = max(np.max(dos_o), np.max(dos_u))
        plot_DOS(ax_occ_dos, energy_o, dos_o,
                 occ_energy_min, occ_energy_max,
                 dos_min, dos_max,
                 flip=False)
        plot_DOS(ax_unocc_dos, energy_u, dos_u,
                 unocc_energy_min, unocc_energy_max,
                 dos_min, dos_max,
                 flip=True)

        return ax_tcm, ax_occ_dos, ax_unocc_dos, ax_spec

    def get_TCM(self, weight_p,
                occ_energy_min, occ_energy_max,
                unocc_energy_min, unocc_energy_max,
                delta_energy, sigma, zero_fermilevel):
        assert weight_p.dtype == float
        u = 0  # TODO

        eig_n = self.eig_un[u].copy()
        if zero_fermilevel:
            eig_n -= self.fermilevel
            fermilevel = 0.0
        else:
            fermilevel = self.fermilevel

        eig_n *= Hartree
        fermilevel *= Hartree

        # Inclusive arange function
        def arange(x0, x1, dx):
            return np.arange(x0, x1 + 0.5 * dx, dx)

        energy_o = arange(occ_energy_min, occ_energy_max, delta_energy)
        energy_u = arange(unocc_energy_min, unocc_energy_max, delta_energy)

        def gauss_ne(energy_e):
            energy_ne = energy_e[np.newaxis, :]
            eig_ne = eig_n[:, np.newaxis]
            norm = 1.0 / (sigma * np.sqrt(2 * np.pi))
            return norm * np.exp(-0.5 * (energy_ne - eig_ne)**2 / sigma**2)

        G_no = gauss_ne(energy_o)
        G_nu = gauss_ne(energy_u)

        # DOS
        dos_o = 2.0 * np.sum(G_no, axis=0)
        dos_u = 2.0 * np.sum(G_nu, axis=0)
        dosmax = max(np.max(dos_o), np.max(dos_u))
        dos_o /= dosmax
        dos_u /= dosmax

        def is_between(x, xmin, xmax):
            return xmin <= x and x <= xmax

        flt_p = []
        buf = 4 * sigma
        for p, weight in enumerate(weight_p):
            i, a = self.ia_p[p]
            if (is_between(eig_n[i],
                           occ_energy_min - buf,
                           occ_energy_max + buf) and
                is_between(eig_n[a],
                           unocc_energy_min - buf,
                           unocc_energy_max + buf)):
                flt_p.append(p)

        weight_f = weight_p[flt_p]
        G_fo = G_no[self.ia_p[flt_p, 0]]
        G_fu = G_nu[self.ia_p[flt_p, 1]]
        G_of = G_fo.T
        tcm_ou = np.dot(G_of * weight_f, G_fu)

        return energy_o, energy_u, dos_o, dos_u, tcm_ou, fermilevel
