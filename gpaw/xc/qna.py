from gpaw.xc.gga import GGA
import numpy as np
from gpaw.lfc import LFC
from gpaw.spline import Spline
from gpaw.xc.gga import gga_x, gga_c


class QNAKernel:
    def __init__(self, qna):
        self.qna = qna
        self.type = 'GGA'
        self.name = 'QNA'
        self.kappa = 0.804

    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg, dedsigma_xg,
                  tau_sg=None, dedtau_sg=None, mu_g=None, beta_g=None,
                  dedmu_g=None, dedbeta_g=None):

        e_g[:] = 0.
        dedsigma_xg[:] = 0.

        if self.qna.override_atoms is not None:
            atoms = self.qna.override_atoms
            self.qna.Pa.set_positions(atoms.get_scaled_positions() % 1.0)
        else:
            atoms = self.qna.atoms

        if len(n_sg.shape) > 2:
            # 3D xc calculation
            mu_g, beta_g = self.qna.calculate_spatial_parameters(atoms)
            dedmu_g = self.qna.dedmu_g
            dedbeta_g = self.qna.dedbeta_g
        else:
            # Atomic xc calculation: use always atomwise mu and beta parameters
            mu, beta = self.qna.parameters[atoms[self.qna.current_atom].symbol]
            mu_g = np.zeros_like(n_sg[0])
            beta_g = np.zeros_like(n_sg[0])
            mu_g[:] = mu
            beta_g[:] = beta
            dedmu_g = None
            dedbeta_g = None

        # Enable to use PBE always
        # mu_g[:] = 0.2195149727645171
        # beta_g[:] = 0.06672455060314922

        # Write mu and beta fields
        if 0:
            from ase.io import write
            write('mu_g.cube', atoms, data=mu_g)
            write('beta_g.cube', atoms, data=beta_g)
            raise SystemExit

        # spin-paired: XXX Copy-paste from gga.py to prevent
        # distruptions to pyGGA
        if len(n_sg) == 1:
            n = n_sg[0]
            n[n < 1e-20] = 1e-40

            # exchange
            res = gga_x(self.name, 0, n, sigma_xg[0], self.kappa, mu_g,
                        dedmu_g=dedmu_g)
            ex, rs, dexdrs, dexda2 = res

            if dedmu_g is not None:
                dedmu_g[:] = n * dedmu_g

            # correlation
            res = gga_c(self.name, 0, n, sigma_xg[0], 0, beta_g,
                        decdbeta_g=dedbeta_g)
            ec, rs_, decdrs, decda2, decdzeta = res
            e_g[:] += n * (ex + ec)
            dedn_sg[:] += ex + ec - rs * (dexdrs + decdrs) / 3.
            dedsigma_xg[:] += n * (dexda2 + decda2)
        # spin-polarized:
        else:
            na = 2. * n_sg[0]
            na[na < 1e-20] = 1e-40

            nb = 2. * n_sg[1]
            nb[nb < 1e-20] = 1e-40

            n = 0.5 * (na + nb)
            zeta = 0.5 * (na - nb) / n

            if dedmu_g is not None:
                dedmua_g = dedmu_g.copy()
                dedmub_g = dedmu_g.copy()
            else:
                dedmua_g = None
                dedmub_g = None

            # exchange
            exa, rsa, dexadrs, dexada2 = gga_x(
                self.name, 1, na, 4.0 * sigma_xg[0], self.kappa, mu_g,
                dedmu_g=dedmua_g)
            exb, rsb, dexbdrs, dexbda2 = gga_x(
                self.name, 1, nb, 4.0 * sigma_xg[2], self.kappa, mu_g,
                dedmu_g=dedmub_g)
            a2 = sigma_xg[0] + 2.0 * sigma_xg[1] + sigma_xg[2]
            if dedmu_g is not None:
                dedmu_g[:] = 0.5 * (na * dedmua_g + nb * dedmub_g)

            # correlation
            ec, rs, decdrs, decda2, decdzeta = gga_c(
                self.name, 1, n, a2, zeta, beta_g, decdbeta_g=dedbeta_g)
            e_g[:] += 0.5 * (na * exa + nb * exb) + n * ec
            dedn_sg[0][:] += (exa + ec - (rsa * dexadrs + rs * decdrs) / 3.0 -
                              (zeta - 1.0) * decdzeta)
            dedn_sg[1][:] += (exb + ec - (rsb * dexbdrs + rs * decdrs) / 3.0 -
                              (zeta + 1.0) * decdzeta)
            dedsigma_xg[0][:] += 2.0 * na * dexada2 + n * decda2
            dedsigma_xg[1][:] += 2.0 * n * decda2
            dedsigma_xg[2][:] += 2.0 * nb * dexbda2 + n * decda2

        if dedbeta_g is not None:
            dedbeta_g[:] = dedbeta_g * n


class QNA(GGA):
    def __init__(self, atoms, parameters, qna_setup_name='PBE', alpha=2.0,
                 override_atoms=None, stencil=2):
        # override_atoms is only used to test the partial derivatives
        # of xc-functional
        kernel = QNAKernel(self)
        GGA.__init__(self, kernel, stencil=stencil)
        self.atoms = atoms
        self.parameters = parameters
        self.qna_setup_name = qna_setup_name
        self.alpha = alpha
        self.override_atoms = override_atoms
        self.orbital_dependent = False

    def todict(self):
        dct = dict(type='qna-gga',
                   name='QNA',
                   setup_name=self.qna_setup_name,
                   parameters=self.parameters,
                   alpha=self.alpha,
                   orbital_dependent=False)
        return dct

    def set_grid_descriptor(self, gd):
        GGA.set_grid_descriptor(self, gd)
        self.dedmu_g = gd.zeros()
        self.dedbeta_g = gd.zeros()
        # Create gaussian LFC
        l_lim = 1.0e-30
        rcut = 12
        points = 200
        r_i = np.linspace(0, rcut, points + 1)
        rcgauss = 1.2
        g_g = (2 / rcgauss**3 / np.pi *
               np.exp(-((r_i / rcgauss)**2)**self.alpha))

        # Values too close to zero can cause numerical problems especially with
        # forces (some parts of the mu and beta field can become negative)
        g_g[np.where(g_g < l_lim)] = l_lim
        spline = Spline(l=0, rmax=rcut, f_g=g_g)
        spline_j = [[spline]] * len(self.atoms)
        self.Pa = LFC(gd, spline_j)

    def set_positions(self, spos_ac, atom_partition=None):
        self.Pa.set_positions(spos_ac)

    def calculate_spatial_parameters(self, atoms):
        mu_g = self.gd.zeros()
        beta_g = self.gd.zeros()
        denominator = self.gd.zeros()
        mu_a = {}
        beta_a = {}
        eye_a = {}
        for atom in atoms:
            mu, beta = self.parameters[atom.symbol]
            mu_a[atom.index] = np.array([mu])
            beta_a[atom.index] = np.array([beta])
            eye_a[atom.index] = np.array(1.0)
        self.Pa.add(mu_g, mu_a)
        self.Pa.add(beta_g, beta_a)
        self.Pa.add(denominator, eye_a)
        mu_g /= denominator
        beta_g /= denominator
        return mu_g, beta_g

    def calculate_paw_correction(self, setup, D_sp, dEdD_sp=None,
                                 addcoredensity=True, a=None):
        self.current_atom = a
        return GGA.calculate_paw_correction(self, setup, D_sp, dEdD_sp,
                                            addcoredensity, a)

    def get_setup_name(self):
        return self.qna_setup_name

    def get_description(self):
        return 'QNA Parameters: ' + str(self.parameters)

    def add_forces(self, F_av):
        mu_g = self.gd.zeros()
        beta_g = self.gd.zeros()
        denominator = self.gd.zeros()
        mu_a = {}
        beta_a = {}
        eye_a = {}
        for atom in self.atoms:
            mu, beta = self.parameters[atom.symbol]
            mu_a[atom.index] = np.array([mu])
            beta_a[atom.index] = np.array([beta])
            eye_a[atom.index] = np.array(1.0)
        self.Pa.add(mu_g, mu_a)
        self.Pa.add(beta_g, beta_a)
        self.Pa.add(denominator, eye_a)
        mu_g /= denominator
        beta_g /= denominator

        # mu
        part1 = -self.dedmu_g / denominator
        part2 = -part1 * mu_g
        c_axiv = self.Pa.dict(derivative=True)
        self.Pa.derivative(part1, c_axiv)

        for atom in self.atoms:
            F_av[atom.index] -= c_axiv[atom.index][0][:] * mu_a[atom.index][0]
        c_axiv = self.Pa.dict(derivative=True)
        self.Pa.derivative(part2, c_axiv)
        for atom in self.atoms:
            F_av[atom.index] -= c_axiv[atom.index][0][:]

        # beta
        part1 = -self.dedbeta_g / denominator
        part2 = -part1 * beta_g
        c_axiv = self.Pa.dict(derivative=True)
        self.Pa.derivative(part1, c_axiv)
        for atom in self.atoms:
            F_av[atom.index] -= c_axiv[atom.index][0] * beta_a[atom.index][0]
        c_axiv = self.Pa.dict(derivative=True)
        self.Pa.derivative(part2, c_axiv)
        for atom in self.atoms:
            F_av[atom.index] -= c_axiv[atom.index][0][:]
