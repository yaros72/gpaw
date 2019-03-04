import numpy as np
import numbers
from scipy.optimize import minimize
from scipy.integrate import simps
from gpaw import GPAW, PW
from ase.parallel import parprint
import scipy.integrate as integrate
from scipy.interpolate import InterpolatedUnivariateSpline
from ase.units import Hartree as Ha


class ElectrostaticCorrections():
    """
    Calculate the electrostatic corrections for charged defects.
    """
    def __init__(self, pristine, charged,
                 q=None, sigma=None, r0=None, dimensionality='3d'):
        if isinstance(pristine, str):
            pristine = GPAW(pristine, txt=None, parallel={'domain': 1})
        if isinstance(charged, str):
            charged = GPAW(charged, txt=None)
        calc = GPAW(mode=PW(500, force_complex_dtype=True),
                    kpts={'size': (1, 1, 1),
                          'gamma': True},
                    parallel={'domain': 1},
                    symmetry='off',
                    txt=None)
        atoms = pristine.atoms.copy()
        calc.initialize(atoms)

        self.pristine = pristine
        self.charged = charged
        self.calc = calc
        self.q = q
        self.sigma = sigma
        self.dimensionality = dimensionality

        self.pd = self.calc.wfs.pd
        self.L = self.pd.gd.cell_cv[2, 2]
        if r0 is not None:
            self.r0 = r0
        elif dimensionality == '2d':
            self.r0 = np.array([0, 0, self.L / 2])
        else:
            self.r0 = np.array([0, 0, 0])
        self.z0 = self.r0[2]

        self.G_Gv = self.pd.get_reciprocal_vectors(q=0,
                                                   add_q=False)  # G in Bohr^-1
        self.G2_G = self.pd.G2_qG[0]  # |\vec{G}|^2 in Bohr^-2
        self.rho_G = self.calculate_gaussian_density()
        self.Omega = abs(np.linalg.det(self.calc.density.gd.cell_cv))
        self.data = None
        self.El = None

        # For the 2D case, we assume that the dielectric profile epsilon(z)
        # follows the electronic density of the pristine system n(z).
        density = self.pristine.get_pseudo_density(gridrefinement=2)
        density_1d = density.mean(axis=(0, 1))
        coarse_z = find_z(self.pristine.density.gd.refine())
        fine_z = find_z(self.pd.gd.refine())
        transformer = InterpolatedUnivariateSpline(coarse_z, density_1d)
        self.density_1d = np.array([transformer(x) for x in fine_z])

        # We potentially have two z axes -- one on the original calculation,
        # and one on the calculator we just set up.
        self.z_g = fine_z
        self.density_z = coarse_z

        self.G_z = find_G_z(self.pd)
        self.G_parallel = np.unique(self.G_Gv[:, :2], axis=0)
        self.GG = np.outer(self.G_z, self.G_z)  # G * Gprime

        # We need the G_z vectors on the finely sampled grid, to calculate
        # epsilon(G - G'), which goes up to 2G_max in magnitude.
        self.G_z_fine = (np.fft.fftfreq(len(fine_z)) *
                         2 * np.pi / (self.L) * len(fine_z))
        self.index_array = self.get_index_array()

        self.epsilons_z = None
        self.Eli = None
        self.Elp = None
        self.epsilon_GG = None

    def calculate_gaussian_density(self):
        # Fourier transformed gaussian:
        return self.q * np.exp(-0.5 * self.G2_G * self.sigma ** 2)

    def set_epsilons(self, epsilons, epsilon_bulk=1):
        """Set the bulk dielectric constant of the system corresponding to the
        atomic configuration of the pristine system. This is used to calculate
        the screened coulomb potential of the gaussian model distribution that
        we use. In 2D, the dielectric constant is poorly defined. However, when
        we do DFT calculations with periodic systems, we are always working
        with pseudo bulk-like structures, due to the periodic boundary
        conditions. It is therefore possible to calculate some dielectric
        constant. The screening of the system must come from where there is an
        electronic density, and we thus set

        epsilon(z) = k * n(z) + epsilon_bulk,

        for some constant of proprotionality k. This is determined by the
        constraint that the total screening int dz epsilon(z) gives the correct
        value.

        For 3d systems, the trick is to set k = 0. In that case, we have the
        constant dielectric function that we require.

        Parameters:
          epsilons: A float, or a list of two floats. If a single
            float is given, the screening in-plane and out-of-plane is assumed
            to be the same. If two floats are given, the first represents the
            in-plane response, and the second represents the out-of-plane
            response.
          epsilon_bulk: The value of the screening far away from the system,
            given in the same format as epsilons. In most cases, this will be
            1, corresponding to vacuum.

        Returns:
            epsilons_z: dictionary containnig the normalized dielectric
              profiles in the in-plane and out-of-plane directions along the z
              axis.
        """

        # Reset the state of the object; the corrections need to be
        # recalculated
        self.Elp = None
        self.Eli = None
        self.epsilon_GG = None

        def normalize(value):
            if isinstance(value, numbers.Number):
                value = [value, value]
            elif len(value) == 1:
                value = [value[0]] * 2
            return np.array(value)

        epsilons = normalize(epsilons)
        self.epsilons = epsilons

        if self.dimensionality == '2d':
            eb = normalize(epsilon_bulk)
        elif self.dimensionality == '3d':
            eb = normalize(epsilons)
        self.eb = eb

        z = self.z_g
        L = self.L
        density_1d = self.density_1d
        N = simps(density_1d, z)
        epsilons_z = np.zeros((2,) + np.shape(density_1d))
        epsilons_z += density_1d

        # In-plane
        epsilons_z[0] = epsilons_z[0] * (epsilons[0] - eb[0]) / N * L + eb[0]

        # Out-of-plane
        def objective_function(k):
            k = k[0]
            integral = simps(1 / (k * density_1d + eb[1]), z) / L
            return np.abs(integral - 1 / epsilons[1])

        test = minimize(objective_function, [1], method='Nelder-Mead')
        assert test.success, "Unable to normalize dielectric profile"
        k = test.x[0]
        epsilons_z[1] = epsilons_z[1] * k + eb[1]
        self.epsilons_z = {'in-plane': epsilons_z[0],
                           'out-of-plane': epsilons_z[1]}
        self.calculate_epsilon_GG()

    def get_index_array(self):
        """
        Calculate the indices that map between the G vectors on the fine grid,
        and the matrix defined by M(G, Gprime) = G - Gprime.

        We use this to generate the matrix epsilon_GG = epsilon(G - Gprime)
        based on knowledge of epsilon(G) on the fine grid.
        """
        G, Gprime = np.meshgrid(self.G_z, self.G_z)
        difference_GG = G - Gprime
        index_array = np.zeros(np.shape(difference_GG))
        index_array[:] = np.nan
        for idx, val in enumerate(self.G_z_fine):
            mask = np.isclose(difference_GG, val)
            index_array[mask] = idx
        if np.isnan(index_array).any():
            print("Missing indices found in mapping between plane wave "
                  "sets. Something is wrong with your G-vectors!")
            print(np.isnan(index_array).all())
            print(np.where(np.isnan(index_array)))
            print(self.G_z_fine)
            print(difference_GG)
            assert False
        return index_array.astype(int)

    def calculate_epsilon_GG(self):
        assert self.epsilons_z is not None, ("You provide the dielectric "
                                             "constant first!")
        N = len(self.density_1d)
        epsilon_par_G = np.fft.fft(self.epsilons_z['in-plane']) / N
        epsilon_perp_G = np.fft.fft(self.epsilons_z['out-of-plane']) / N
        epsilon_par_GG = epsilon_par_G[self.index_array]
        epsilon_perp_GG = epsilon_perp_G[self.index_array]

        self.epsilon_GG = {'in-plane': epsilon_par_GG,
                           'out-of-plane': epsilon_perp_GG}

    def calculate_periodic_correction(self):
        G_z = self.G_z
        if self.Elp is not None:
            return self.Elp
        if self.epsilon_GG is None:
            self.calculate_epsilon_GG()
        Elp = 0.0

        for vector in self.G_parallel:
            G2 = (np.dot(vector, vector) + G_z * G_z)
            rho_G = self.q * np.exp(- G2 * self.sigma ** 2 / 2, dtype=complex)
            if self.dimensionality == '2d':
                rho_G *= np.exp(1j * self.z0 * G_z)
            A_GG = (self.GG * self.epsilon_GG['out-of-plane']
                    + np.dot(vector, vector) * self.epsilon_GG['in-plane'])
            if np.allclose(vector, 0):
                A_GG[0, 0] = 1  # The d.c. potential is poorly defined
            V_G = np.linalg.solve(A_GG, rho_G)
            if np.allclose(vector, 0):
                parprint('Skipping G^2=0 contribution to Elp')
                V_G[0] = 0
            Elp += (rho_G * V_G).sum()

        Elp *= 2.0 * np.pi * Ha / self.Omega

        self.Elp = Elp
        return Elp

    def calculate_isolated_correction(self):
        if self.Eli is not None:
            return self.Eli

        L = self.L
        G_z = self.G_z

        G, Gprime = np.meshgrid(G_z, G_z)
        Delta_GG = G - Gprime
        phase = Delta_GG * self.z0
        gaussian = (Gprime ** 2 + G ** 2) * self.sigma * self.sigma / 2

        prefactor = np.exp(1j * phase - gaussian)

        if self.epsilon_GG is None:
            self.calculate_epsilon_GG()

        dE_GG_par = (self.epsilon_GG['in-plane']
                     - self.eb[0] * np.eye(len(self.G_z)))
        dE_GG_perp = (self.epsilon_GG['out-of-plane']
                      - self.eb[1] * np.eye(len(self.G_z)))

        def integrand(k):
            K_inv_G = ((self.eb[0] * k ** 2 + self.eb[1] * G_z ** 2) /
                       (1 - np.exp(-k * L / 2) * np.cos(L * G_z / 2)))
            K_inv_GG = np.diag(K_inv_G)
            D_GG = L * (K_inv_GG + dE_GG_perp * self.GG + dE_GG_par * k ** 2)
            return (k * np.exp(-k ** 2 * self.sigma ** 2) *
                    (prefactor * np.linalg.inv(D_GG)).sum())

        I = integrate.quad(integrand, 0, np.inf, limit=500)
        Eli = self.q * self.q * I[0] * Ha
        self.Eli = Eli
        return Eli

    def calculate_potential_alignment(self):
        V_neutral = -np.mean(self.pristine.get_electrostatic_potential(),
                             (0, 1))
        V_charged = -np.mean(self.charged.get_electrostatic_potential(),
                             (0, 1))

        z = self.density_z
        z_model, V_model = self.calculate_z_avg_model_potential()
        V_model = InterpolatedUnivariateSpline(z_model[:-1], V_model[:-1])
        V_model = V_model(z)
        self.V_model = V_model
        Delta_V = self.average(V_model - V_charged + V_neutral, z)
        return Delta_V

    def calculate_z_avg_model_potential(self):

        vox3 = self.calc.density.gd.cell_cv[2, :] / len(self.z_g)

        # The grid is arranged with z increasing fastest, then y
        # then x (like a cube file)

        G_z = self.G_z[1:]
        rho_Gz = self.q * np.exp(-0.5 * G_z * G_z * self.sigma * self.sigma)

        zs = []
        Vs = []
        if self.dimensionality == '2d':
            phase = np.exp(1j * (self.G_z * self.z0))
            A_GG = (self.GG * self.epsilon_GG['out-of-plane'])
            A_GG[0, 0] = 1
            V_G = np.linalg.solve(A_GG,
                                  phase * np.array([0] + list(rho_Gz)))[1:]
        elif self.dimensionality == '3d':
            V_G = rho_Gz / self.eb[1] / G_z ** 2

        for z in self.z_g:
            phase_G = np.exp(1j * (G_z * z))
            V = (np.sum(phase_G * V_G)
                 * Ha * 4.0 * np.pi / (self.Omega))
            Vs.append(V)

        V = (np.sum(V_G) * Ha * 4.0 * np.pi / (self.Omega))
        zs = list(self.z_g) + [vox3[2]]
        Vs.append(V)
        return np.array(zs), np.array(Vs)

    def average(self, V, z):
        N = len(V)
        if self.dimensionality == '3d':
            middle = N // 2
            points = range(middle - N // 8, middle + N // 8 + 1)
            restricted = V[points]
        elif self.dimensionality == '2d':
            points = list(range(0, N // 8)) + list(range(7 * N // 8, N))
        restricted = V[points]
        V_mean = np.mean(restricted)
        return V_mean

    def calculate_corrected_formation_energy(self):
        E_0 = self.pristine.get_potential_energy()
        E_X = self.charged.get_potential_energy()
        Eli = self.calculate_isolated_correction().real
        Elp = self.calculate_periodic_correction().real
        Delta_V = self.calculate_potential_alignment()
        return E_X - E_0 - (Elp - Eli) + Delta_V * self.q

    def calculate_uncorrected_formation_energy(self):
        E_0 = self.pristine.get_potential_energy()
        E_X = self.charged.get_potential_energy()
        return E_X - E_0

    def collect_electrostatic_data(self):
        V_neutral = -np.mean(self.pristine.get_electrostatic_potential(),
                             (0, 1)),
        V_charged = -np.mean(self.charged.get_electrostatic_potential(),
                             (0, 1)),
        data = {'epsilon': self.eb[0],
                'z': self.density_z,
                'V_0': V_neutral,
                'V_X': V_charged,
                'Elc': self.Elp - self.Eli,
                'D_V_mean': self.calculate_potential_alignment(),
                'V_model': self.V_model,
                'D_V': (self.V_model
                        - V_neutral
                        + V_charged)}
        self.data = data
        return data


def find_G_z(pd):
    G_Gv = pd.get_reciprocal_vectors(q=0)
    mask = (G_Gv[:, 0] == 0) & (G_Gv[:, 1] == 0)
    G_z = G_Gv[mask][:, 2]  # G_z vectors in Bohr^{-1}
    return G_z


def find_z(gd):
    r3_xyz = gd.get_grid_point_coordinates()
    nrz = r3_xyz.shape[3]
    return r3_xyz[2].flatten()[:nrz]
