from __future__ import print_function

import pickle
import numpy as np
from math import pi
import ase.units
from ase.parallel import world
from ase.utils import devnull
import sys
import os

Hartree = ase.units.Hartree
Bohr = ase.units.Bohr


def load(fd):
    try:
        return pickle.load(fd, encoding='latin1')
    except TypeError:
        return pickle.load(fd)


class Heterostructure:
    """This class defines dielectric function of heterostructures
        and related physical quantities."""
    def __init__(self, structure, d,
                 include_dipole=True, d0=None,
                 wmax=10, qmax=None):
        """Creates a Heterostructure object.

        structure: list of str
            Heterostructure set up. Each entry should consist of number of
            layers + chemical formula.
            For example: ['3H-MoS2', graphene', '10H-WS2'] gives 3 layers of
            H-MoS2, 1 layer of graphene and 10 layers of H-WS2.
        d: array of floats
            Interlayer distances for neighboring layers in Ang.
            Length of array = number of layers - 1
        d0: float
            The width of a single layer in Ang, only used for monolayer
            calculation. The layer separation in bulk is typically a good
            measure.
        include_dipole: Bool
            Includes dipole contribution if True
        wmax: float
            Cutoff for frequency grid (eV)
        qmax: float
            Cutoff for wave-vector grid (1/Ang)
        """

        chi_monopole = []
        drho_monopole = []
        if include_dipole:
            chi_dipole = []
            drho_dipole = []
        else:
            self.chi_dipole = None
            drho_dipole = None
        self.z = []
        layer_indices = []
        self.n_layers = 0
        namelist = []
        n_rep = 0
        structure = expand_layers(structure)

        if not check_building_blocks(list(set(structure))):
            raise ValueError('Building Blocks not on the same grid')
        self.n_layers = len(structure)
        for n, name in enumerate(structure):
            if name not in namelist:
                namelist.append(name)
                data = np.load(name + '-chi.npz')
                q = data['q_abs']
                w = data['omega_w']
                zi = data['z']
                zi -= np.mean(zi)
                chim = data['chiM_qw']
                chid = data['chiD_qw']
                drhom = data['drhoM_qz']
                drhod = data['drhoD_qz']
                if qmax is not None:
                    qindex = np.argmin(abs(q - qmax * Bohr)) + 1
                else:
                    qindex = None
                if wmax is not None:
                    windex = np.argmin(abs(w - wmax / Hartree)) + 1
                else:
                    windex = None
                chi_monopole.append(np.array(chim[:qindex, :windex]))
                drho_monopole.append(np.array(drhom[:qindex]))
                if include_dipole:
                    chi_dipole.append(np.array(chid[:qindex, :windex]))
                    drho_dipole.append(np.array(drhod[:qindex]))
                self.z.append(np.array(zi))
                n -= n_rep
            else:
                n = namelist.index(name)
                n_rep += 1

            layer_indices = np.append(layer_indices, n)

        self.layer_indices = np.array(layer_indices, dtype=int)

        self.q_abs = q[:qindex]
        if self.q_abs[0] == 0:
            self.q_abs[0] += 1e-12

        # parallelize over q in case of multiple processors
        from ase.parallel import world
        self.world = world
        nq = len(self.q_abs)
        mynq = (nq + self.world.size - 1) // self.world.size
        self.q1 = min(mynq * self.world.rank, nq)
        self.q2 = min(self.q1 + mynq, nq)
        self.mynq = self.q2 - self.q1
        self.myq_abs = self.q_abs[self.q1: self.q2]

        self.frequencies = w[:windex]
        self.n_types = len(namelist)

        chi_monopole = np.array(chi_monopole)[:, self.q1: self.q2]
        chi_dipole = np.array(chi_dipole)[:, self.q1: self.q2]
        for i in range(self.n_types):
            drho_monopole[i] = np.array(drho_monopole[i])[self.q1: self.q2]
            drho_dipole[i] = np.array(drho_dipole[i])[self.q1: self.q2]

        # layers and distances
        self.d = np.array(d) / Bohr  # interlayer distances
        if self.n_layers > 1:
            # space around each layer
            self.s = (np.insert(self.d, 0, self.d[0]) +
                      np.append(self.d, self.d[-1])) / 2.
        else:  # Monolayer calculation
            self.s = [d0 / Bohr]  # Width of single layer

        self.dim = np.copy(self.n_layers)
        if include_dipole:
            self.dim *= 2

        # Grid stuff
        self.poisson_lim = 100  # above this limit use potential model
        edgesize = 50
        system_size = np.sum(self.d) + edgesize
        self.z_lim = system_size
        self.dz = 0.01
        # master grid
        self.z_big = np.arange(0, self.z_lim, self.dz) - edgesize / 2.
        self.z0 = np.append(np.array([0]), np.cumsum(self.d))

        # arange potential and density
        self.chi_monopole = chi_monopole
        if include_dipole:
            self.chi_dipole = chi_dipole
        self.drho_monopole, self.drho_dipole, self.basis_array, \
            self.drho_array = self.arange_basis(drho_monopole, drho_dipole)

        self.dphi_array = self.get_induced_potentials()
        self.kernel_qij = None

    def arange_basis(self, drhom, drhod=None):
        from scipy.interpolate import interp1d
        Nz = len(self.z_big)
        drho_array = np.zeros([self.dim, self.mynq,
                               Nz], dtype=complex)
        basis_array = np.zeros([self.dim, self.mynq,
                                Nz], dtype=complex)

        for i in range(self.n_types):
            z = self.z[i]
            drhom_i = drhom[i]
            fm = interp1d(z, np.real(drhom_i))
            fm2 = interp1d(z, np.imag(drhom_i))
            if drhod is not None:
                drhod_i = drhod[i]
                fd = interp1d(z, np.real(drhod_i))
                fd2 = interp1d(z, np.imag(drhod_i))
            for k in [k for k in range(self.n_layers)
                      if self.layer_indices[k] == i]:
                z_big = self.z_big - self.z0[k]
                i_1s = np.argmin(np.abs(-self.s[k] / 2. - z_big))
                i_2s = np.argmin(np.abs(self.s[k] / 2. - z_big))

                i_1 = np.argmin(np.abs(z[0] - z_big))
                if z_big[i_1] < z[0]:
                    i_1 += 1
                i_2 = np.argmin(np.abs(z[-1] - z_big))
                if z_big[i_2] > z[-1]:
                    i_2 -= 1
                if drhod is not None:
                    drho_array[2 * k, :, i_1: i_2 + 1] = \
                        fm(z_big[i_1: i_2 + 1]) + 1j * fm2(z_big[i_1: i_2 + 1])
                    basis_array[2 * k, :, i_1s: i_2s] = 1. / self.s[k]
                    drho_array[2 * k + 1, :, i_1: i_2 + 1] = \
                        fd(z_big[i_1: i_2 + 1]) + 1j * fd2(z_big[i_1: i_2 + 1])
                    basis_array[2 * k + 1, :, i_1: i_2] = \
                        fd(z_big[i_1: i_2]) + 1j * fd2(z_big[i_1: i_2])
                else:
                    drho_array[k, :, i_1: i_2 + 1] = \
                        fm(z_big[i_1: i_2 + 1]) + 1j * fm2(z_big[i_1: i_2 + 1])
                    basis_array[k, :, i_1s:i_2s] = 1. / self.s[k]

        return drhom, drhod, basis_array, drho_array

    def get_induced_potentials(self):
        from scipy.interpolate import interp1d
        Nz = len(self.z_big)
        dphi_array = np.zeros([self.dim, self.mynq, Nz], dtype=complex)

        for i in range(self.n_types):
            z = self.z[i]
            for iq in range(self.mynq):
                q = self.myq_abs[iq]
                drho_m = self.drho_monopole[i][iq].copy()
                poisson_m = self.solve_poisson_1D(drho_m, q, z)
                z_poisson = self.get_z_grid(z, z_lim=self.poisson_lim)
                if not len(z_poisson) == len(np.real(poisson_m)):
                    z_poisson = z_poisson[:len(poisson_m)]
                    poisson_m = poisson_m[:len(z_poisson)]
                fm = interp1d(z_poisson, np.real(poisson_m))
                fm2 = interp1d(z_poisson, np.imag(poisson_m))
                if self.chi_dipole is not None:
                    drho_d = self.drho_dipole[i][iq].copy()
                    #  delta is the distance between dipole peaks / 2
                    delta = np.abs(z[np.argmax(drho_d)] -
                                   z[np.argmin(drho_d)]) / 2.
                    poisson_d = self.solve_poisson_1D(drho_d, q, z,
                                                      dipole=True,
                                                      delta=delta)
                    fd = interp1d(z_poisson, np.real(poisson_d))
                    fd2 = interp1d(z_poisson, np.imag(poisson_d))

                for k in [k for k in range(self.n_layers)
                          if self.layer_indices[k] == i]:
                    z_big = self.z_big - self.z0[k]
                    i_1 = np.argmin(np.abs(z_poisson[0] - z_big))
                    if z_big[i_1] < z_poisson[0]:
                        i_1 += 1
                    i_2 = np.argmin(np.abs(z_poisson[-1] - z_big))
                    if z_big[i_2] > z_poisson[-1]:
                        i_2 -= 1

                    dphi_array[self.dim // self.n_layers * k, iq] = (
                        self.potential_model(self.myq_abs[iq], self.z_big,
                                             self.z0[k]))
                    dphi_array[self.dim // self.n_layers * k,
                               iq, i_1: i_2 + 1] = (fm(z_big[i_1: i_2 + 1]) +
                                                1j * fm2(z_big[i_1: i_2 + 1]))
                    if self.chi_dipole is not None:
                        dphi_array[2 * k + 1, iq] = \
                            self.potential_model(self.myq_abs[iq], self.z_big,
                                                 self.z0[k], dipole=True,
                                                 delta=delta)
                        dphi_array[2 * k + 1, iq, i_1: i_2 + 1] = \
                            fd(z_big[i_1: i_2 + 1]) + 1j * fd2(z_big[i_1: i_2 + 1])

        return dphi_array

    def get_z_grid(self, z, z_lim=None):
        dz = z[1] - z[0]
        if z_lim is None:
            z_lim = self.z_lim

        z_lim = int(z_lim / dz) * dz
        z_grid = np.insert(z, 0, np.flip(np.arange(z[0] - dz, -z_lim - dz, -dz)))
        z_grid = np.append(z_grid, np.arange(z[-1] + dz, z_lim + dz, dz))
        return z_grid

    def potential_model(self, q, z, z0=0, dipole=False, delta=None):
        """
        2D Coulomb: 2 pi / q with exponential decay in z-direction
        """
        if dipole:  # Two planes separated by 2 * delta
            V = np.pi / (q * delta) * \
                (-np.exp(-q * np.abs(z - z0 + delta)) +
                 np.exp(-q * np.abs(z - z0 - delta)))
        else:  # Monopole potential from single plane
            V = 2 * np.pi / q * np.exp(-q * np.abs(z - z0))

        return V

    def solve_poisson_1D(self, drho, q, z,
                         dipole=False, delta=None):
        """
        Solves poissons equation in 1D using finite difference method.

        drho: induced potential basis function
        q: momentum transfer.
        """
        dz = z[1] - z[0]
        z_grid = self.get_z_grid(z, z_lim=self.poisson_lim)

        assert np.allclose(np.diff(z_grid), dz)
        Nz_loc = (len(z_grid) - len(z)) // 2

        drho = np.append(np.insert(drho, 0, np.zeros([Nz_loc])),
                         np.zeros([Nz_loc]))
        Nint = len(drho) - 1

        bc_v0 = self.potential_model(q, z_grid[0], dipole=dipole,
                                     delta=delta)
        bc_vN = self.potential_model(q, z_grid[-1], dipole=dipole,
                                     delta=delta)
        M = np.zeros((Nint + 1, Nint + 1))
        f_z = np.zeros(Nint + 1, dtype=complex)
        f_z[:] = - 4 * np.pi * drho[:]
        # Finite Difference Matrix
        for i in range(1, Nint):
            M[i, i] = -2. / (dz**2) - q**2
            M[i, i + 1] = 1. / dz**2
            M[i, i - 1] = 1. / dz**2
            M[0, 0] = 1.
            M[Nint, Nint] = 1.

        f_z[0] = bc_v0
        f_z[Nint] = bc_vN

        # Getting the Potential
        M_inv = np.linalg.inv(M)
        dphi = np.dot(M_inv, f_z)
        return dphi

    def get_Coulomb_Kernel(self, step_potential=False):
        kernel_qij = np.zeros([self.mynq, self.dim, self.dim],
                              dtype=complex)
        for iq in range(self.mynq):
            if np.isclose(self.myq_abs[iq], 0):
                # Special treatment of q=0 limit
                kernel_qij[iq] = np.eye(self.dim)
            else:
                if step_potential:
                    # Use step-function average for monopole contribution
                    kernel_qij[iq] = np.dot(self.basis_array[:, iq],
                                            self.dphi_array[:, iq].T) * self.dz
                else:  # Normal kernel
                    kernel_qij[iq] = np.dot(self.drho_array[:, iq],
                                            self.dphi_array[:, iq].T) * self.dz

        return kernel_qij

    def get_chi_matrix(self):

        """
        Dyson equation: chi_full = chi_intra + chi_intra V_inter chi_full
        """
        Nls = self.n_layers
        q_abs = self.myq_abs
        chi_m_iqw = self.chi_monopole
        chi_d_iqw = self.chi_dipole

        if self.kernel_qij is None:
            self.kernel_qij = self.get_Coulomb_Kernel()
        chi_qwij = np.zeros((self.mynq, len(self.frequencies),
                             self.dim, self.dim), dtype=complex)

        for iq in range(len(q_abs)):
            kernel_ij = self.kernel_qij[iq].copy()
            np.fill_diagonal(kernel_ij, 0)  # Diagonal is set to zero
            for iw in range(0, len(self.frequencies)):
                chi_intra_i = chi_m_iqw[self.layer_indices, iq, iw]
                if self.chi_dipole is not None:
                    chi_intra_i = np.insert(chi_intra_i, np.arange(Nls) + 1,
                                            chi_d_iqw[self.layer_indices,
                                                      iq, iw])
                chi_intra_ij = np.diag(chi_intra_i)
                chi_qwij[iq, iw, :, :] = np.dot(
                    np.linalg.inv(np.eye(self.dim) -
                                  np.dot(chi_intra_ij, kernel_ij)),
                    chi_intra_ij)

        return chi_qwij

    def get_eps_matrix(self, step_potential=False):
        """
        Get dielectric matrix as: eps^{-1} = 1 + V chi_full
        """
        self.kernel_qij =\
            self.get_Coulomb_Kernel(step_potential=step_potential)

        chi_qwij = self.get_chi_matrix()
        eps_qwij = np.zeros((self.mynq, len(self.frequencies),
                             self.dim, self.dim), dtype=complex)

        for iq in range(self.mynq):
            kernel_ij = self.kernel_qij[iq]
            for iw in range(0, len(self.frequencies)):
                eps_qwij[iq, iw, :, :] = np.linalg.inv(
                    np.eye(kernel_ij.shape[0]) + np.dot(kernel_ij,
                                                        chi_qwij[iq, iw,
                                                                 :, :]))

        return eps_qwij

    def get_screened_potential(self, layer=0):
        """
        get the screened interaction averaged over layer "k":
        W_{kk}(q, w) = \sum_{ij} V_{ki}(q) \chi_{ij}(q, w) V_{jk}(q)

        parameters:
        layer: int
            index of layer to calculate the screened interaction for.

        returns: W(q,w)
        """
        self.kernel_qij =\
            self.get_Coulomb_Kernel(step_potential=True)

        chi_qwij = self.get_chi_matrix()
        W_qw = np.zeros((self.mynq, len(self.frequencies)),
                        dtype=complex)

        W0_qw = np.zeros((self.mynq, len(self.frequencies)),
                         dtype=complex)
        k = layer
        if self.chi_dipole is not None:
            k *= 2
        for iq in range(self.mynq):
            kernel_ij = self.kernel_qij[iq]
            if np.isclose(self.myq_abs[iq], 0):
                kernel_ij = 2 * np.pi * np.ones([self.dim, self.dim])

            if self.chi_dipole is not None:
                for j in range(self.n_layers):
                    kernel_ij[2 * j, 2 * j + 1] = 0
                    kernel_ij[2 * j + 1, 2 * j] = 0

            for iw in range(0, len(self.frequencies)):
                W_qw[iq, iw] = np.dot(np.dot(kernel_ij[k], chi_qwij[iq, iw]),
                                      kernel_ij[:, k])
                W0_qw[iq, iw] = kernel_ij[k, k]**2 * chi_qwij[iq, iw, k, k]

        W_qw = self.collect_qw(W_qw)

        return W_qw

    def get_exciton_screened_potential(self, e_distr, h_distr):
        v_screened_q = np.zeros(self.mynq)
        eps_qwij = self.get_eps_matrix()
        h_distr = h_distr.transpose()
        kernel_qij = self.get_Coulomb_Kernel()

        for iq in range(0, self.mynq):
            ext_pot = np.dot(kernel_qij[iq], h_distr)
            v_screened_q[iq] =\
                np.dot(e_distr,
                       np.dot(np.linalg.inv(eps_qwij[iq, 0, :, :]),
                              ext_pot)).real

        v_screened_q = self.collect_q(v_screened_q[:])

        return self.q_abs, -v_screened_q, kernel_qij

    def get_exciton_screened_potential_r(self, r_array, e_distr=None,
                                         h_distr=None, Wq_name=None,
                                         tweak=None):
        if Wq_name is not None:
            q_abs, W_q = load(open(Wq_name, 'rb'))
        else:
            q_temp, W_q, xxx = self.get_exciton_screened_potential(e_distr,
                                                                   h_distr)

        from scipy.special import jn

        inter = False
        if np.where(e_distr == 1)[0][0] != np.where(h_distr == 1)[0][0]:
            inter = True

        layer_dist = 0.
        if self.n_layers == 1:
            layer_thickness = self.s[0]
        else:
            if len(e_distr) == self.n_layers:
                div = 1
            else:
                div = 2

            if not inter:
                ilayer = np.where(e_distr == 1)[0][0] // div
                if ilayer == len(self.d):
                    ilayer -= 1
                layer_thickness = self.d[ilayer]
            else:
                ilayer1 = np.min([np.where(e_distr == 1)[0][0],
                                  np.where(h_distr == 1)[0][0]]) // div
                ilayer2 = np.max([np.where(e_distr == 1)[0][0],
                                  np.where(h_distr == 1)[0][0]]) // div
                layer_thickness = self.d[ilayer1] / 2.
                layer_dist = np.sum(self.d[ilayer1:ilayer2]) / 1.8
        if tweak is not None:
            layer_thickness = tweak

        W_q *= q_temp
        q = np.linspace(q_temp[0], q_temp[-1], 10000)
        Wt_q = np.interp(q, q_temp, W_q)
        Dq_Q2D = q[1] - q[0]

        if not inter:
            Coulombt_q = (-4. * np.pi / q *
                          (1. - np.exp(-q * layer_thickness / 2.)) /
                          layer_thickness)
        else:
            Coulombt_q = (-2 * np.pi / q *
                          (np.exp(-q * (layer_dist - layer_thickness / 2.)) -
                           np.exp(-q * (layer_dist + layer_thickness / 2.))) /
                          layer_thickness)

        W_r = np.zeros(len(r_array))
        for ir in range(len(r_array)):
            J_q = jn(0, q * r_array[ir])
            if r_array[ir] > np.exp(-13):
                Int_temp = (
                    -1. / layer_thickness *
                    np.log((layer_thickness / 2. - layer_dist +
                            np.sqrt(r_array[ir]**2 +
                                    (layer_thickness / 2. - layer_dist)**2)) /
                           (-layer_thickness / 2. - layer_dist +
                            np.sqrt(r_array[ir]**2 +
                                    (layer_thickness / 2. + layer_dist)**2))))
            else:
                if not inter:
                    Int_temp = (-1. / layer_thickness *
                                np.log(layer_thickness**2 / r_array[ir]**2))
                else:
                    Int_temp = (-1. / layer_thickness *
                                np.log((layer_thickness + 2 * layer_dist) /
                                       (2 * layer_dist - layer_thickness)))

            W_r[ir] = (Dq_Q2D / 2. / np.pi *
                       np.sum(J_q * (Wt_q - Coulombt_q)) +
                       Int_temp)

        return r_array, W_r

    def get_exciton_binding_energies(self, eff_mass, L_min=-50, L_max=10,
                                     Delta=0.1, e_distr=None, h_distr=None,
                                     Wq_name=None, tweak=None):
        from scipy.linalg import eig
        r_space = np.arange(L_min, L_max, Delta)
        Nint = len(r_space)

        r, W_r = self.get_exciton_screened_potential_r(r_array=np.exp(r_space),
                                                       e_distr=e_distr,
                                                       h_distr=h_distr,
                                                       Wq_name=None,
                                                       tweak=tweak)

        H = np.zeros((Nint, Nint), dtype=complex)
        for i in range(0, Nint):
            r_abs = np.exp(r_space[i])
            H[i, i] = - 1. / r_abs**2 / 2. / eff_mass \
                * (-2. / Delta**2 + 1. / 4.) + W_r[i]
            if i + 1 < Nint:
                H[i, i + 1] = -1. / r_abs**2 / 2. / eff_mass \
                    * (1. / Delta**2 - 1. / 2. / Delta)
            if i - 1 >= 0:
                H[i, i - 1] = -1. / r_abs**2 / 2. / eff_mass \
                    * (1. / Delta**2 + 1. / 2. / Delta)

        ee, ev = eig(H)
        index_sort = np.argsort(ee.real)
        ee = ee[index_sort]
        ev = ev[:, index_sort]
        return ee * Hartree, ev

    def get_macroscopic_dielectric_function(self, static=True, layers=None,
                                            direction='x'):
        """
        Calculates the averaged dielectric function over the structure.

        Parameters:

        static: bool
            If True only include w=0

        layers: array of integers
            list with index of specific layers to include in the average.

        direction: str 'x' or 'z'
            'x' for in plane dielectric function
            'z' for out of plane dielectric function

        Returns list of q-points, frequencies, dielectric function(q, w).
        """
        layer_weight = self.s / np.sum(self.s) * self.n_layers

        if self.chi_dipole is not None:
            layer_weight = np.insert(layer_weight,
                                     np.arange(self.n_layers) + 1,
                                     layer_weight)

        if direction == 'x':
            const_per = np.ones([self.n_layers])
            if self.chi_dipole is not None:
                const_per = np.insert(const_per, np.arange(self.n_layers) + 1,
                                      np.zeros([self.n_layers]))

        elif direction == 'z':
            const_per = np.zeros([self.n_layers])
            assert self.chi_dipole is not None
            const_per = np.insert(const_per, np.arange(self.n_layers) + 1,
                                  np.ones([self.n_layers]))

        if layers is None:  # average over entire structure
            N = self.n_layers
            potential = const_per
        else:  # average over selected layers
            N = len(layers)
            potential = np.zeros([self.dim])
            index = layers * self.dim / self.n_layers
            if direction == 'z':
                index += 1
            potential[index] = 1.

        if static:
            Nw = 1
        else:
            Nw = len(self.frequencies)

        eps_qwij = self.get_eps_matrix(step_potential=True)[:, :Nw]

        Nq = self.mynq
        epsM_qw = np.zeros([Nq, Nw], dtype=complex)

        for iw in range(Nw):
            for iq in range(Nq):
                eps_ij = eps_qwij[iq, iw]
                epsinv_ij = np.linalg.inv(eps_ij)
                epsinvM = 1. / N * np.dot(np.array(potential) * layer_weight,
                                          np.dot(epsinv_ij,
                                                 np.array(const_per)))

                epsM_qw[iq, iw] = 1. / epsinvM

        epsM_qw = self.collect_qw(epsM_qw)

        return self.q_abs / Bohr, self.frequencies[:Nw] * Hartree, epsM_qw

    def get_eels(self, dipole_contribution=False):
        """
        Calculates Electron energy loss spectrum, defined as:

        EELS(q, w) = - Im 4 \pi / q**2 \chiM(q, w)

        Returns list of q-points, Frequencies and the loss function
        """
        const_per = np.ones([self.n_layers])
        layer_weight = self.s / np.sum(self.s) * self.n_layers

        if self.chi_dipole is not None:
            const_per = np.insert(const_per,
                                  np.arange(self.n_layers) + 1,
                                  np.zeros([self.n_layers]))
            layer_weight = np.insert(layer_weight,
                                     np.arange(self.n_layers) + 1,
                                     layer_weight)

        if dipole_contribution:
            const_per = np.zeros([self.n_layers])
            const_per = np.insert(const_per,
                                  np.arange(self.n_layers) + 1,
                                  np.ones([self.n_layers]))

        N = self.n_layers
        eels_qw = np.zeros([self.mynq, len(self.frequencies)],
                           dtype=complex)

        chi_qwij = self.get_chi_matrix()

        for iq in range(self.mynq):
            for iw in range(len(self.frequencies)):
                eels_qw[iq, iw] = np.dot(np.array(const_per) * layer_weight,
                                         np.dot(chi_qwij[iq, iw],
                                                np.array(const_per)))

            eels_qw[iq, :] *= 1. / N * 4 * np.pi / self.q_abs[iq]**2

        eels_qw = self.collect_qw(eels_qw)

        return self.q_abs / Bohr, self.frequencies * Hartree, \
            - (Bohr * eels_qw).imag

    def get_absorption_spectrum(self, dipole_contribution=False):
        """
        Calculates absorption spectrum, defined as:

        ABS(q, w) = - Im 2 / q \epsM(q, w)

        Returns list of q-points, Frequencies and the loss function
        """
        const_per = np.ones([self.n_layers])
        layer_weight = self.s / np.sum(self.s) * self.n_layers

        if self.chi_dipole is not None:
            const_per = np.insert(const_per,
                                  np.arange(self.n_layers) + 1,
                                  np.zeros([self.n_layers]))
            layer_weight = np.insert(layer_weight,
                                     np.arange(self.n_layers) + 1,
                                     layer_weight)

        if dipole_contribution:
            const_per = np.zeros([self.n_layers])
            const_per = np.insert(const_per,
                                  np.arange(self.n_layers) + 1,
                                  np.ones([self.n_layers]))

        N = self.n_layers
        abs_qw = np.zeros([self.mynq, len(self.frequencies)],
                          dtype=complex)

        eps_qwij = self.get_eps_matrix()

        for iq in range(self.mynq):
            for iw in range(len(self.frequencies)):
                abs_qw[iq, iw] = np.dot(np.array(const_per) * layer_weight,
                                        np.dot(eps_qwij[iq, iw],
                                               np.array(const_per)))

            abs_qw[iq, :] *= 1. / N * 2. / self.q_abs[iq]

        abs_qw = self.collect_qw(abs_qw)
        return self.q_abs / Bohr, self.frequencies * Hartree, \
            (Bohr * abs_qw).imag

    def get_sum_eels(self, V_beam=100, include_z=False):

        """
        Calculates the q- averaged Electron energy loss spectrum usually
        obtained in scanning transmission electron microscopy (TEM).

        EELS(w) = - Im [sum_{q}^{q_max}  V(q) \chi(w, q) V(q)]
                    \delta(w - q \dot v_e)

        The calculation assumes a beam in the z-direction perpendicular to the
        layers, and that the response in isotropic within the plane.

        Input parameters:
        V_beam: float
            Acceleration voltage of electron beam in kV.
            Is used to calculate v_e that goes into \delta(w - q \dot v_e)

        Returns list of Frequencies and the loss function
        """
        const_per = np.ones([self.n_layers])
        layer_weight = self.s / np.sum(self.s) * self.n_layers

        if self.chi_dipole is not None:
            const_per = np.insert(const_per,
                                  np.arange(self.n_layers) + 1,
                                  np.zeros([self.n_layers]))
            layer_weight = np.insert(layer_weight,
                                     np.arange(self.n_layers) + 1,
                                     layer_weight)

        eels_w = np.zeros([len(self.frequencies)], dtype=complex)
        chi_qwij = self.get_chi_matrix()
        vol = np.pi * (self.q_abs[-1] + self.q_abs[1] / 2.)**2
        weight0 = np.pi * (self.q_abs[1] / 2.)**2 / vol
        c = (1 - weight0) / np.sum(self.q_abs)
        weights = c * self.q_abs
        weights[0] = weight0
        # Beam speed from relativistic eq
        me = ase.units._me
        c = ase.units._c
        E_0 = me * c**2  # Rest energy
        E = E_0 + V_beam * 1e3 / ase.units.J   # Relativistic energy
        v_e = c * (E**2 - E_0**2)**0.5 / E  # beam velocity in SI
        # Lower cutoff q_z = w / v_e
        w_wSI = self.frequencies * Hartree \
            / ase.units.J / ase.units._hbar  # w in SI units
        q_z = w_wSI / v_e / ase.units.m * Bohr  # in Bohr
        q_z[0] = q_z[1]
        print('Using a beam acceleration voltage of V = %3.1f kV' % (V_beam))
        print('Beam speed = %1.2f / c' % (v_e / c))
        # Upper cutoff q_c = q[1] / 2.
        q_c = self.q_abs[1] / 2.
        # Integral for q=0: \int_0^q_c \frac{q^3}{(q^2 + q_z^2)^2} dq
        I = 2 * np.pi / vol * \
            (q_z**2 / 2. / (q_c**2 + q_z**2) - 0.5 +
             0.5 * np.log((q_c / q_z)**2 + 1))
        I2 = 2 * np.pi / vol / 2. * (1. / q_z**2 - 1. / (q_z**2 + q_c**2))

        for iq in range(self.mynq):
            eels_temp = np.zeros([len(self.frequencies)], dtype=complex)
            for iw in range(len(self.frequencies)):
                # Longitudinal in-plane
                temp = np.dot(np.array(const_per) * layer_weight,
                              np.dot(chi_qwij[iq, iw], np.array(const_per)))
                eels_temp[iw] += temp

            if np.isclose(self.q_abs[iq], 0):
                eels_temp *= (4 * np.pi)**2 * I

            else:
                eels_temp *= 1. / (self.q_abs[iq]**2 + q_z**2)**2
                eels_temp *= (4 * np.pi)**2 * weights[iq]
            eels_w += eels_temp

            if include_z:
                eels_temp = np.zeros([len(self.frequencies)], dtype=complex)
                for iw in range(len(self.frequencies)):
                    # longitudinal out of plane
                    temp = np.dot(np.array(const_per[::-1]) * layer_weight,
                                  np.dot(chi_qwij[iq, iw],
                                         np.array(const_per[::-1])))
                    eels_temp[iw] += temp

                    # longitudinal cross terms
                    temp = 1J * np.dot(np.array(const_per) * layer_weight,
                                       np.dot(chi_qwij[iq, iw],
                                              np.array(const_per[::-1])))
                    eels_temp[iw] += temp / q_z[iw]

                    temp = -1J * np.dot(np.array(const_per[::-1]) *
                                        layer_weight,
                                        np.dot(chi_qwij[iq, iw],
                                               np.array(const_per)))
                    eels_temp[iw] += temp / q_z[iw]

                    # Transversal
                    temp = np.dot(np.array(const_per[::-1]) * layer_weight,
                                  np.dot(chi_qwij[iq, iw],
                                         np.array(const_per[::-1])))
                    temp *= (v_e / c)**4
                    eels_temp[iw] += temp

                if np.isclose(self.q_abs[iq], 0):
                    eels_temp *= (4 * np.pi)**2 * I2 * q_z**2
                else:
                    eels_temp *= 1. / (self.q_abs[iq]**2 + q_z**2)**2 * q_z**2
                    eels_temp *= (4 * np.pi)**2 * weights[iq]

                eels_w += eels_temp

        self.world.sum(eels_w)

        return self.frequencies * Hartree, - (Bohr**5 * eels_w * vol).imag

    def get_response(self, iw=0, dipole=False):
        """
        Get the induced density and potential due to constant perturbation
        obtained as: rho_ind(r) = \int chi(r,r') dr'
        """
        const_per = np.ones([self.n_layers])
        if self.chi_dipole is not None:
            const_per = np.insert(const_per,
                                  np.arange(self.n_layers) + 1,
                                  np.zeros([self.n_layers]))

        if dipole:
            const_per = self.z0 - self.z0[-1] / 2.
            const_per = np.insert(const_per,
                                  np.arange(self.n_layers) + 1,
                                  np.ones([self.n_layers]))

        chi_qij = self.get_chi_matrix()[:, iw]
        Vind_qz = np.zeros((self.mynq, len(self.z_big)))
        rhoind_qz = np.zeros((self.mynq, len(self.z_big)))

        drho_array = self.drho_array.copy()
        dphi_array = self.dphi_array.copy()
        # Expand on potential and density basis function
        # to get spatial detendence
        for iq in range(self.mynq):
            chi_ij = chi_qij[iq]
            Vind_qi = np.dot(chi_ij, np.array(const_per))
            rhoind_qz[iq] = np.dot(drho_array[:, iq].T, Vind_qi)
            Vind_qz[iq] = np.dot(dphi_array[:, iq].T, Vind_qi)

        rhoind_qz = self.collect_qw(rhoind_qz)
        return self.z_big * Bohr, rhoind_qz, Vind_qz, self.z0 * Bohr

    def get_plasmon_eigenmodes(self):
        """
        Diagonalize the dieletric matrix to get the plasmon eigenresonances
        of the system.

        Returns:
            Eigenvalue array (shape Nq x nw x dim), z-grid, induced densities,
            induced potentials, energies at zero crossings.
        """

        assert self.world.size == 1
        eps_qwij = self.get_eps_matrix()
        Nw = len(self.frequencies)
        Nq = self.mynq
        w_w = self.frequencies
        eig = np.zeros([Nq, Nw, self.dim], dtype=complex)
        vec = np.zeros([Nq, Nw, self.dim, self.dim],
                       dtype=complex)

        omega0 = [[] for i in range(Nq)]
        rho_z = [np.zeros([0, len(self.z_big)]) for i in range(Nq)]
        phi_z = [np.zeros([0, len(self.z_big)]) for i in range(Nq)]
        for iq in range(Nq):
            m = 0
            eig[iq, 0], vec[iq, 0] = np.linalg.eig(eps_qwij[iq, 0])
            vec_dual = np.linalg.inv(vec[iq, 0])
            for iw in range(1, Nw):
                eig[iq, iw], vec_p = np.linalg.eig(eps_qwij[iq, iw])
                vec_dual_p = np.linalg.inv(vec_p)
                overlap = np.abs(np.dot(vec_dual, vec_p))
                index = list(np.argsort(overlap)[:, -1])
                if len(np.unique(index)) < self.dim:  # add missing indices
                    addlist = []
                    removelist = []
                    for j in range(self.dim):
                        if index.count(j) < 1:
                            addlist.append(j)
                        if index.count(j) > 1:
                            for l in range(1, index.count(j)):
                                removelist.append(
                                    np.argwhere(np.array(index) == j)[l])
                    for j in range(len(addlist)):
                        index[removelist[j][0]] = addlist[j]
                vec[iq, iw] = vec_p[:, index]
                vec_dual = vec_dual_p[index, :]
                eig[iq, iw, :] = eig[iq, iw, index]
                klist = [k for k in range(self.dim)
                         if (eig[iq, iw - 1, k] < 0 and eig[iq, iw, k] > 0)]
                for k in klist:  # Eigenvalue crossing
                    a = np.real((eig[iq, iw, k] - eig[iq, iw - 1, k]) /
                                (w_w[iw] - w_w[iw - 1]))
                    # linear interp for crossing point
                    w0 = np.real(-eig[iq, iw - 1, k]) / a + w_w[iw - 1]
                    rho = np.dot(self.drho_array[:, iq, :].T, vec_dual[k, :])
                    phi = np.dot(self.dphi_array[:, iq, :].T,
                                 vec[iq, iw, :, k])
                    rho_z[iq] = np.append(rho_z[iq], rho[np.newaxis, :],
                                          axis=0)
                    phi_z[iq] = np.append(phi_z[iq], phi[np.newaxis, :],
                                          axis=0)
                    omega0[iq].append(w0)
                    m += 1

        return eig, self.z_big * Bohr, rho_z, phi_z, np.array(omega0)

    def collect_q(self, a_q):
        """ Collect arrays of dim (q)"""
        world = self.world
        nq = len(self.q_abs)
        mynq = (nq + self.world.size - 1) // self.world.size
        b_q = np.zeros(mynq, a_q.dtype)
        b_q[:self.q2 - self.q1] = a_q
        A_q = np.empty(mynq * world.size, a_q.dtype)
        if world.size == 1:
            A_q[:] = b_q
        else:
            world.all_gather(b_q, A_q)
        return A_q[:nq]

    def collect_qw(self, a_qw):
        """ Collect arrays of dim (q X w)"""
        nw = a_qw.shape[1]
        nq = len(self.q_abs)
        A_qw = np.zeros((nq, nw),
                        a_qw.dtype)
        for iw in range(nw):
            A_qw[:, iw] = self.collect_q(a_qw[:, iw])
        nq = len(self.q_abs)
        return A_qw[:nq]


class BuildingBlock():

    """ Module for using Linear response to calculate dielectric
    building block of 2D material with GPAW"""

    def __init__(self, filename, df, isotropic_q=True, nq_inf=10,
                 direction='x', qmax=None, txt=sys.stdout):
        """Creates a BuildingBlock object.

        filename: str
            used to save data file: filename-chi.npz
        df: DielectricFunction object
            Determines how linear response calculation is performed
        isotropic_q: bool
            If True, only q-points along one direction (1 0 0) in the
            2D BZ is included, thus asuming an isotropic material
        direction: 'x' or 'y'
            Direction used for isotropic q sampling.
        qmax: float
            Cutoff for q-grid. To be used if one wishes to sample outside the
            irreducible BZ. Only works for isotropic q-sampling.
        nq_inf: int
            number of extra q points in the limit q->0 along each direction,
            extrapolated from q=0, assumung that the head of chi0_wGG goes
            as q^2 and the wings as q.
            Note that this does not hold for (semi)metals!

        """
        if qmax is not None:
            assert isotropic_q
        self.filename = filename
        self.isotropic_q = isotropic_q
        self.nq_inf = nq_inf
        self.nq_inftot = nq_inf
        if not isotropic_q:
            self.nq_inftot *= 2

        if direction == 'x':
            qdir = 0
        elif direction == 'y':
            qdir = 1
        self.direction = direction

        self.df = df  # dielectric function object
        self.df.truncation = '2D'  # in case you forgot!
        self.omega_w = self.df.chi0.omega_w
        self.world = self.df.chi0.world

        if self.world.rank != 0:
            txt = devnull
        elif isinstance(txt, str):
            txt = open(txt, 'w', 1)
        self.fd = txt

        calc = self.df.chi0.calc
        kd = calc.wfs.kd
        self.kd = kd
        r = calc.wfs.gd.get_grid_point_coordinates()
        self.z = r[2, 0, 0, :]

        nw = self.omega_w.shape[0]
        self.chiM_qw = np.zeros([0, nw])
        self.chiD_qw = np.zeros([0, nw])
        self.drhoM_qz = np.zeros([0, self.z.shape[0]])
        self.drhoD_qz = np.zeros([0, self.z.shape[0]])

        # First: choose all ibzq in 2D BZ
        from ase.dft.kpoints import monkhorst_pack
        from gpaw.kpt_descriptor import KPointDescriptor
        offset_c = 0.5 * ((kd.N_c + 1) % 2) / kd.N_c
        bzq_qc = monkhorst_pack(kd.N_c) + offset_c
        qd = KPointDescriptor(bzq_qc)
        qd.set_symmetry(calc.atoms, kd.symmetry)
        q_cs = qd.ibzk_kc
        rcell_cv = 2 * pi * np.linalg.inv(calc.wfs.gd.cell_cv).T
        if isotropic_q:  # only use q along [1 0 0] or [0 1 0] direction.
            Nk = kd.N_c[qdir]
            qx = np.array(range(1, Nk // 2)) / float(Nk)
            q_cs = np.zeros([Nk // 2 - 1, 3])
            q_cs[:, qdir] = qx
            q = 0
            if qmax is not None:
                qmax *= Bohr
                qmax_v = np.zeros([3])
                qmax_v[qdir] = qmax
                q_c = q_cs[-1]
                q_v = np.dot(q_c, rcell_cv)
                q = (q_v**2).sum()**0.5
                assert Nk % 2 == 0
                i = Nk / 2.0
                while q < qmax:
                    if i == Nk:  # omit BZ edge
                        i += 1
                        continue
                    q_c = np.zeros([3])
                    q_c[qdir] = i / Nk
                    q_cs = np.append(q_cs, q_c[np.newaxis, :], axis=0)
                    q_v = np.dot(q_c, rcell_cv)
                    q = (q_v**2).sum()**0.5
                    i += 1
        q_vs = np.dot(q_cs, rcell_cv)
        q_abs = (q_vs**2).sum(axis=1)**0.5
        sort = np.argsort(q_abs)
        q_abs = q_abs[sort]
        q_cs = q_cs[sort]
        if isotropic_q:
            q_cut = q_abs[0] / 2  # extrapolate to half of smallest finite q
        else:
            q_cut = q_abs[1]  # smallest finite q
        self.nq_cut = self.nq_inftot + 1

        q_infs = np.zeros([q_cs.shape[0] + self.nq_inftot, 3])
        # x-direction:
        q_infs[: self.nq_inftot, qdir] = \
            np.linspace(1e-05, q_cut, self.nq_inftot + 1)[:-1]
        if not isotropic_q:  # y-direction
            q_infs[self.nq_inf:self.nq_inftot + 1, 1] = \
                np.linspace(0, q_cut, self.nq_inf + 1)[1:]

        # add q_inf to list
        self.q_cs = np.insert(q_cs, 0, np.zeros([self.nq_inftot, 3]), axis=0)
        self.q_vs = np.dot(self.q_cs, rcell_cv)
        self.q_vs += q_infs
        self.q_abs = (self.q_vs**2).sum(axis=1)**0.5
        self.q_infs = q_infs
        self.complete = False
        self.nq = 0
        if self.load_chi_file():
            if self.complete:
                print('Building block loaded from file', file=self.fd)
        world.barrier()

    def calculate_building_block(self, add_intraband=False):
        if self.complete:
            return
        Nq = self.q_cs.shape[0]
        for nq in range(self.nq, Nq):
            self.nq = nq
            self.save_chi_file()
            q_c = self.q_cs[nq]
            q_inf = self.q_infs[nq]
            if np.allclose(q_inf, 0):
                q_inf = None

            qcstr = '(' + ', '.join(['%.3f' % x for x in q_c]) + ')'
            print('Calculating contribution from q-point #%d/%d, q_c=%s'
                  % (nq + 1, Nq, qcstr), file=self.fd)
            if q_inf is not None:
                qstr = '(' + ', '.join(['%.3f' % x for x in q_inf]) + ')'
                print('    and q_inf=%s' % qstr, file=self.fd)
            pd, chi0_wGG, \
                chi_wGG = self.df.get_dielectric_matrix(
                    symmetric=False,
                    calculate_chi=True,
                    q_c=q_c,
                    q_v=q_inf,
                    direction=self.direction,
                    add_intraband=add_intraband)
            print('calculated chi!', file=self.fd)

            nw = len(self.omega_w)
            world = self.df.chi0.world
            w1 = min(self.df.mynw * world.rank, nw)

            q, omega_w, chiM_qw, chiD_qw, z, drhoM_qz, drhoD_qz = \
                get_chi_2D(self.omega_w, pd, chi_wGG)

            chiM_w = chiM_qw[0]
            chiD_w = chiD_qw[0]
            chiM_w = self.collect(chiM_w)
            chiD_w = self.collect(chiD_w)

            if self.world.rank == 0:
                assert w1 == 0  # drhoM and drhoD in static limit
                self.update_building_block(chiM_w[np.newaxis, :],
                                           chiD_w[np.newaxis, :],
                                           drhoM_qz, drhoD_qz)

        # Induced densities are not probably described in q-> 0 limit-
        # replace with finite q result:
        if self.world.rank == 0:
            for n in range(Nq):
                if np.allclose(self.q_cs[n], 0):
                    self.drhoM_qz[n] = self.drhoM_qz[self.nq_cut]
                    self.drhoD_qz[n] = self.drhoD_qz[self.nq_cut]

        self.complete = True
        self.save_chi_file()

        return

    def update_building_block(self, chiM_qw, chiD_qw, drhoM_qz,
                              drhoD_qz):

        self.chiM_qw = np.append(self.chiM_qw, chiM_qw, axis=0)
        self.chiD_qw = np.append(self.chiD_qw, chiD_qw, axis=0)
        self.drhoM_qz = np.append(self.drhoM_qz, drhoM_qz, axis=0)
        self.drhoD_qz = np.append(self.drhoD_qz, drhoD_qz, axis=0)

    def save_chi_file(self, filename=None):
        if filename is None:
            filename = self.filename
        data = {'last_q': self.nq,
                'complete': self.complete,
                'isotropic_q': self.isotropic_q,
                'q_cs': self.q_cs,
                'q_vs': self.q_vs,
                'q_abs': self.q_abs,
                'omega_w': self.omega_w,
                'chiM_qw': self.chiM_qw,
                'chiD_qw': self.chiD_qw,
                'z': self.z,
                'drhoM_qz': self.drhoM_qz,
                'drhoD_qz': self.drhoD_qz}

        if self.world.rank == 0:
            np.savez_compressed(filename + '-chi.npz',
                                **data)
        world.barrier()

    def load_chi_file(self):
        try:
            data = np.load(self.filename + '-chi.npz')
        except IOError:
            return False
        if (np.all(data['omega_w'] == self.omega_w) and
            np.all(data['q_cs'] == self.q_cs) and
            np.all(data['z'] == self.z)):
            self.nq = data['last_q']
            self.complete = data['complete']
            self.chiM_qw = data['chiM_qw']
            self.chiD_qw = data['chiD_qw']
            self.drhoM_qz = data['drhoM_qz']
            self.drhoD_qz = data['drhoD_qz']
            return True
        else:
            return False

    def interpolate_to_grid(self, q_grid, w_grid):

        """
        Parameters
        q_grid: in Ang. should start at q=0
        w_grid: in eV
        """

        from scipy.interpolate import RectBivariateSpline
        from scipy.interpolate import interp1d
        if not self.complete:
            self.calculate_building_block()
        q_grid *= Bohr
        w_grid /= Hartree

        assert np.max(q_grid) <= np.max(self.q_abs), \
            'q can not be larger that %1.2f Ang' % np.max(self.q_abs / Bohr)
        assert np.max(w_grid) <= np.max(self.omega_w), \
            'w can not be larger that %1.2f eV' % \
            np.max(self.omega_w * Hartree)

        sort = np.argsort(self.q_abs)
        q_abs = self.q_abs[sort]

        # chi monopole
        self.chiM_qw = self.chiM_qw[sort]

        omit_q0 = False
        if np.isclose(q_abs[0], 0) and not np.isclose(self.chiM_qw[0, 0], 0):
            omit_q0 = True  # omit q=0 from interpolation
            q0_abs = q_abs[0].copy()
            q_abs[0] = 0.
            chi0_w = self.chiM_qw[0].copy()
            self.chiM_qw[0] = np.zeros_like(chi0_w)

        yr = RectBivariateSpline(q_abs, self.omega_w,
                                 self.chiM_qw.real,
                                 s=0)

        yi = RectBivariateSpline(q_abs, self.omega_w,
                                 self.chiM_qw.imag, s=0)

        self.chiM_qw = yr(q_grid, w_grid) + 1j * yi(q_grid, w_grid)
        if omit_q0:
            yr = interp1d(self.omega_w, chi0_w.real)
            yi = interp1d(self.omega_w, chi0_w.imag)
            chi0_w = yr(w_grid) + 1j * yi(w_grid)
            q_abs[0] = q0_abs
            if np.isclose(q_grid[0], 0):
                self.chiM_qw[0] = chi0_w

        # chi dipole
        yr = RectBivariateSpline(q_abs, self.omega_w,
                                 self.chiD_qw[sort].real,
                                 s=0)
        yi = RectBivariateSpline(q_abs, self.omega_w,
                                 self.chiD_qw[sort].imag,
                                 s=0)

        self.chiD_qw = yr(q_grid, w_grid) + 1j * yi(q_grid, w_grid)

        # drho monopole

        yr = RectBivariateSpline(q_abs, self.z,
                                 self.drhoM_qz[sort].real, s=0)
        yi = RectBivariateSpline(q_abs, self.z,
                                 self.drhoM_qz[sort].imag, s=0)

        self.drhoM_qz = yr(q_grid, self.z) + 1j * yi(q_grid, self.z)

        # drho dipole
        yr = RectBivariateSpline(q_abs, self.z,
                                 self.drhoD_qz[sort].real, s=0)
        yi = RectBivariateSpline(q_abs, self.z,
                                 self.drhoD_qz[sort].imag, s=0)

        self.drhoD_qz = yr(q_grid, self.z) + 1j * yi(q_grid, self.z)

        self.q_abs = q_grid
        self.omega_w = w_grid

        self.save_chi_file(filename=self.filename + '_int')

    def collect(self, a_w):
        world = self.df.chi0.world
        b_w = np.zeros(self.df.mynw, a_w.dtype)
        b_w[:self.df.w2 - self.df.w1] = a_w
        nw = len(self.omega_w)
        A_w = np.empty(world.size * self.df.mynw, a_w.dtype)
        world.all_gather(b_w, A_w)
        return A_w[:nw]

    def clear_temp_files(self):
        if not self.savechi0:
            world = self.df.chi0.world
            if world.rank == 0:
                while len(self.temp_files) > 0:
                    filename = self.temp_files.pop()
                    os.remove(filename)


"""TOOLS"""


def check_building_blocks(BBfiles=None):
    """ Check that building blocks are on same frequency-
    and q- grid.

    BBfiles: list of str
        list of names of BB files
    """
    name = BBfiles[0] + '-chi.npz'
    data = np.load(name)
    try:
        q = data['q_abs'].copy()
        w = data['omega_w'].copy()
    except TypeError:
        # Skip test for old format:
        return True
    for name in BBfiles[1:]:
        data = np.load(name + '-chi.npz')
        if not ((data['q_abs'] == q).all and
                (data['omega_w'] == w).all):
            return False
    return True


def interpolate_building_blocks(BBfiles=None, BBmotherfile=None,
                                q_grid=None, w_grid=None):
    """ Interpolate building blocks to same frequency-
    and q- grid

    BBfiles: list of str
        list of names of BB files to be interpolated
    BBmother: str
        name of BB file to match the grids to. Will
        also be interpolated to common grid.
    q_grid: float
        q-grid in Ang. Should start at q=0
    w_grid: float
        in eV
    """

    from scipy.interpolate import RectBivariateSpline, interp1d

    if BBmotherfile is not None:
        BBfiles.append(BBmotherfile)

    q_max = 1000
    w_max = 1000
    for name in BBfiles:
        data = np.load(open(name + '-chi.npz', 'rb'))
        q_abs = data['q_abs']
        q_max = np.min([q_abs[-1], q_max])
        ow = data['omega_w']
        w_max = np.min([ow[-1], w_max])

    if BBmotherfile is not None:
        data = np.load(BBmotherfile + "-chi.npz")
        q_grid = data['q_abs']
        w_grid = data['omega_w']
    else:
        q_grid = q_grid * Bohr
        w_grid = w_grid / Hartree

    q_grid = [q for q in q_grid if q < q_max]
    q_grid.append(q_max)
    w_grid = [w for w in w_grid if w < w_max]
    w_grid.append(w_max)
    q_grid = np.array(q_grid)
    w_grid = np.array(w_grid)
    for name in BBfiles:
        assert data['isotropic_q']
        data = np.load(name + '-chi.npz')
        q_abs = data['q_abs']
        w = data['omega_w']
        z = data['z']
        chiM_qw = data['chiM_qw']
        chiD_qw = data['chiD_qw']
        drhoM_qz = data['drhoM_qz']
        drhoD_qz = data['drhoD_qz']

        # chi monopole
        omit_q0 = False
        if np.isclose(q_abs[0], 0) and not np.isclose(chiM_qw[0, 0], 0):
            omit_q0 = True  # omit q=0 from interpolation
            q0_abs = q_abs[0].copy()
            q_abs[0] = 0.
            chi0_w = chiM_qw[0].copy()
            chiM_qw[0] = np.zeros_like(chi0_w)

        yr = RectBivariateSpline(q_abs, w,
                                 chiM_qw.real,
                                 s=0)

        yi = RectBivariateSpline(q_abs, w,
                                 chiM_qw.imag, s=0)

        chiM_qw = yr(q_grid, w_grid) + 1j * yi(q_grid, w_grid)

        if omit_q0:
            yr = interp1d(w, chi0_w.real)
            yi = interp1d(w, chi0_w.imag)
            chi0_w = yr(w_grid) + 1j * yi(w_grid)
            q_abs[0] = q0_abs
            if np.isclose(q_grid[0], 0):
                chiM_qw[0] = chi0_w

        # chi dipole
        yr = RectBivariateSpline(q_abs, w,
                                 chiD_qw.real,
                                 s=0)
        yi = RectBivariateSpline(q_abs, w,
                                 chiD_qw.imag,
                                 s=0)

        chiD_qw = yr(q_grid, w_grid) + 1j * yi(q_grid, w_grid)

        # drho monopole

        yr = RectBivariateSpline(q_abs, z,
                                 drhoM_qz.real, s=0)
        yi = RectBivariateSpline(q_abs, z,
                                 drhoM_qz.imag, s=0)

        drhoM_qz = yr(q_grid, z) + 1j * yi(q_grid, z)

        # drho dipole
        yr = RectBivariateSpline(q_abs, z,
                                 drhoD_qz.real, s=0)
        yi = RectBivariateSpline(q_abs, z,
                                 drhoD_qz.imag, s=0)

        drhoD_qz = yr(q_grid, z) + 1j * yi(q_grid, z)

        q_abs = q_grid
        omega_w = w_grid

        data = {'q_abs': q_abs,
                'omega_w': omega_w,
                'chiM_qw': chiM_qw,
                'chiD_qw': chiD_qw,
                'z': z,
                'drhoM_qz': drhoM_qz,
                'drhoD_qz': drhoD_qz,
                'isotropic_q': True}

        np.savez_compressed(name + "_int-chi.npz",
                            **data)


def get_chi_2D(omega_w=None, pd=None, chi_wGG=None, q0=None,
               filenames=None, name=None):
    """Calculate the monopole and dipole contribution to the
    2D susceptibillity chi_2D, defined as

    ::

      \chi^M_2D(q, \omega) = \int\int dr dr' \chi(q, \omega, r,r') \\
                          = L \chi_{G=G'=0}(q, \omega)
      \chi^D_2D(q, \omega) = \int\int dr dr' z \chi(q, \omega, r,r') z'
                           = 1/L sum_{G_z,G_z'} z_factor(G_z)
                           chi_{G_z,G_z'} z_factor(G_z'),
      Where z_factor(G_z) =  +/- i e^{+/- i*G_z*z0}
      (L G_z cos(G_z L/2)-2 sin(G_z L/2))/G_z^2

    input parameters:

    filenames: list of str
        list of chi_wGG.pckl files for different q calculated with
        the DielectricFunction module in GPAW
    name: str
        name writing output files
    """

    q_list_abs = []
    if chi_wGG is None and filenames is not None:
        omega_w, pd, chi_wGG, q0 = read_chi_wGG(filenames[0])
        nq = len(filenames)
    elif chi_wGG is not None:
        nq = 1
    nw = chi_wGG.shape[0]
    r = pd.gd.get_grid_point_coordinates()
    z = r[2, 0, 0, :]
    L = pd.gd.cell_cv[2, 2]  # Length of cell in Bohr
    z0 = L / 2.  # position of layer
    chiM_qw = np.zeros([nq, nw], dtype=complex)
    chiD_qw = np.zeros([nq, nw], dtype=complex)
    drhoM_qz = np.zeros([nq, len(z)], dtype=complex)  # induced density
    drhoD_qz = np.zeros([nq, len(z)], dtype=complex)  # induced dipole density
    for iq in range(nq):
        if iq != 0:
            omega_w, pd, chi_wGG, q0 = read_chi_wGG(filenames[iq])
        if q0 is not None:
            q = q0
        else:
            q = pd.K_qv
        npw = chi_wGG.shape[1]
        Gvec = pd.get_reciprocal_vectors(add_q=False)

        Glist = []
        for iG in range(npw):  # List of G with Gx,Gy = 0
            if Gvec[iG, 0] == 0 and Gvec[iG, 1] == 0:
                Glist.append(iG)

        chiM_qw[iq] = L * chi_wGG[:, 0, 0]
        drhoM_qz[iq] += chi_wGG[0, 0, 0]
        q_abs = np.linalg.norm(q)
        q_list_abs.append(q_abs)
        for iG in Glist[1:]:
            G_z = Gvec[iG, 2]
            qGr_R = np.inner(G_z, z.T).T
            # Fourier transform to get induced density at \omega=0
            drhoM_qz[iq] += np.exp(1j * qGr_R) * chi_wGG[0, iG, 0]
            for iG1 in Glist[1:]:
                G_z1 = Gvec[iG1, 2]
                # integrate with z along both coordinates
                factor = z_factor(z0, L, G_z)
                factor1 = z_factor(z0, L, G_z1, sign=-1)
                chiD_qw[iq, :] += 1. / L * factor * chi_wGG[:, iG, iG1] * \
                    factor1
                # induced dipole density due to V_ext = z
                drhoD_qz[iq, :] += 1. / L * np.exp(1j * qGr_R) * \
                    chi_wGG[0, iG, iG1] * factor1
    # Normalize induced densities with chi
    drhoM_qz /= np.repeat(chiM_qw[:, 0, np.newaxis], drhoM_qz.shape[1],
                          axis=1)
    drhoD_qz /= np.repeat(chiD_qw[:, 0, np.newaxis], drhoM_qz.shape[1],
                          axis=1)

    """ Returns q array, frequency array, chi2D monopole and dipole, induced
    densities and z array (all in Bohr)
    """
    if name is not None:
        arrays = [np.array(q_list_abs), omega_w, chiM_qw, chiD_qw,
                  z, drhoM_qz, drhoD_qz]
        names = ['q_abs', 'omega_w', 'z',
                 'chiM_qw', 'chiD_qw',
                 'drhoM_qz', 'drhoD_qz']
        data = {}
        for array, name in zip(arrays, names):
            data[name] = array
        np.save(name + '-chi.npz', **data)
    return np.array(q_list_abs) / Bohr, omega_w * Hartree, chiM_qw, \
        chiD_qw, z, drhoM_qz, drhoD_qz


def z_factor(z0, d, G, sign=1):
    factor = -1j * sign * np.exp(1j * sign * G * z0) * \
        (d * G * np.cos(G * d / 2.) - 2. * np.sin(G * d / 2.)) / G**2
    return factor


def z_factor2(z0, d, G, sign=1):
    factor = sign * np.exp(1j * sign * G * z0) * np.sin(G * d / 2.)
    return factor


def expand_layers(structure):
    newlist = []
    for name in structure:
        num = ''
        while name[0].isdigit():
            num += name[0]
            name = name[1:]
        try:
            num = int(num)
        except ValueError:
            num = 1
        for n in range(num):
            newlist.append(name)
    return newlist


def read_chi_wGG(name):
    """
    Read density response matrix calculated with the DielectricFunction
    module in GPAW.
    Returns frequency grid, gpaw.wavefunctions object, chi_wGG
    """
    fd = open(name, 'rb')
    omega_w, pd, chi_wGG, q0, chi0_wvv = load(fd)
    nw = len(omega_w)
    nG = pd.ngmax
    chi_wGG = np.empty((nw, nG, nG), complex)
    for chi_GG in chi_wGG:
        chi_GG[:] = load(fd)
    return omega_w, pd, chi_wGG, q0
