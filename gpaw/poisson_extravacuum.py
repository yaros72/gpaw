import numpy as np

from ase.units import Bohr
from ase.geometry import cell_to_cellpar

from gpaw.transformers import Transformer
from gpaw.fd_operators import Laplace
from gpaw.poisson import _PoissonSolver
from gpaw.poisson import create_poisson_solver

from gpaw.utilities.extend_grid import extended_grid_descriptor, \
    extend_array, deextend_array


def ext_gd(gd, **kwargs):
    egd, _, _ = extended_grid_descriptor(gd, **kwargs)
    return egd


class ExtraVacuumPoissonSolver(_PoissonSolver):
    """Wrapper around PoissonSolver extending the vacuum size.

       Poisson equation is solved on the large grid defined by `gpts`
       using `poissonsolver_large`.

       If `coarses` is given, then the large grid is first coarsed
       `coarses` times from to the original grid. Then, the coarse
       potential is used to correct the boundary conditions
       of the potential calculated on the original (small, fine)
       grid by `poissonsolver_small`.

       The parameters `nn_*` control the finite difference stencils
       used in the coarsening, refining, and Laplace.

       If the parameter `use_aux_grid` is `True`, an auxiliary
       medium-sized grid is used between the large and small grids.
       The parameter does not affect the result but can be used to
       achieve speed-up depending on the grid sizes.
       """

    def __init__(self, gpts, poissonsolver_large,
                 poissonsolver_small=None, coarses=0,
                 nn_coarse=3, nn_refine=3, nn_laplace=3,
                 use_aux_grid=True):
        # TODO: Alternative options: vacuum size and h
        self.N_large_fine_c = np.array(gpts, dtype=int)
        self.Ncoar = coarses  # coar == coarse
        if self.Ncoar > 0:
            self.use_coarse = True
        else:
            self.use_coarse = False
        self.ps_large_coar = create_poisson_solver(poissonsolver_large)
        if self.use_coarse:
            self.ps_small_fine = create_poisson_solver(poissonsolver_small)
        else:
            assert poissonsolver_small is None
        self.nn_coarse = nn_coarse
        self.nn_refine = nn_refine
        self.nn_laplace = nn_laplace
        self.use_aux_grid = use_aux_grid
        self._initialized = False

    def set_grid_descriptor(self, gd):
        # If non-periodic boundary conditions is used,
        # there is problems with auxiliary grid.
        # Maybe with use_aux_grid=False it would work?
        if gd.pbc_c.any():
            raise NotImplementedError('Only non-periodic boundary '
                                      'conditions are tested')

        self.gd_small_fine = gd
        assert np.all(self.gd_small_fine.N_c <= self.N_large_fine_c), \
            'extended grid has to be larger than the original one'

        if self.use_coarse:
            # 1.1. Construct coarse chain on the small grid
            self.coarser_i = []
            gd = self.gd_small_fine
            N_c = self.N_large_fine_c.copy()
            for i in range(self.Ncoar):
                gd2 = gd.coarsen()
                self.coarser_i.append(Transformer(gd, gd2, self.nn_coarse))
                N_c //= 2
                gd = gd2
            self.gd_small_coar = gd
        else:
            self.gd_small_coar = self.gd_small_fine
            N_c = self.N_large_fine_c

        # 1.2. Construct coarse extended grid
        self.gd_large_coar = ext_gd(self.gd_small_coar, N_c=N_c)

        # Initialize poissonsolvers
        self.ps_large_coar.set_grid_descriptor(self.gd_large_coar)
        if not self.use_coarse:
            return
        self.ps_small_fine.set_grid_descriptor(self.gd_small_fine)

        if self.use_aux_grid:
            # 2.1. Construct an auxiliary grid that is the small grid plus
            # a buffer region allowing Laplace and refining
            # with the used stencils
            buf = self.nn_refine
            for i in range(self.Ncoar):
                buf = 2 * buf + self.nn_refine
            buf += self.nn_laplace
            div = 2**self.Ncoar
            if buf % div != 0:
                buf += div - buf % div
            N_c = self.gd_small_fine.N_c + 2 * buf
            if np.any(N_c > self.N_large_fine_c):
                self.use_aux_grid = False
                N_c = self.N_large_fine_c
            self.gd_aux_fine = ext_gd(self.gd_small_fine, N_c=N_c)
        else:
            self.gd_aux_fine = ext_gd(self.gd_small_fine,
                                      N_c=self.N_large_fine_c)

        # 2.2 Construct Laplace on the aux grid
        self.laplace_aux_fine = Laplace(self.gd_aux_fine, - 0.25 / np.pi,
                                        self.nn_laplace)

        # 2.3 Construct refine chain
        self.refiner_i = []
        gd = self.gd_aux_fine
        N_c = gd.N_c.copy()
        for i in range(self.Ncoar):
            gd2 = gd.coarsen()
            self.refiner_i.append(Transformer(gd2, gd, self.nn_refine))
            N_c //= 2
            gd = gd2
        self.refiner_i = self.refiner_i[::-1]
        self.gd_aux_coar = gd

        if self.use_aux_grid:
            # 2.4 Construct large coarse grid from aux grid
            self.gd_large_coar_from_aux = ext_gd(self.gd_aux_coar,
                                                 N_c=self.gd_large_coar.N_c)
            # Check the consistency of the grids
            gd1 = self.gd_large_coar
            gd2 = self.gd_large_coar_from_aux
            assert np.all(gd1.N_c == gd2.N_c) and np.all(gd1.h_cv == gd2.h_cv)

        self._initialized = False

    def _init(self):
        if self._initialized:
            return
        # Allocate arrays
        self.phi_large_coar_g = self.gd_large_coar.zeros()
        self._initialized = True

        # Initialize poissonsolvers
        #self.ps_large_coar._init()
        #if not self.use_coarse:
        #    return
        #self.ps_small_fine._init()

    def solve(self, phi, rho, **kwargs):
        self._init()
        phi_small_fine_g = phi
        rho_small_fine_g = rho.copy()

        if self.use_coarse:
            # 1.1. Coarse rho on the small grid
            tmp_g = rho_small_fine_g
            for coarser in self.coarser_i:
                tmp_g = coarser.apply(tmp_g)
            rho_small_coar_g = tmp_g
        else:
            rho_small_coar_g = rho_small_fine_g

        # 1.2. Extend rho to the large grid
        rho_large_coar_g = self.gd_large_coar.zeros()
        extend_array(self.gd_small_coar, self.gd_large_coar,
                     rho_small_coar_g, rho_large_coar_g)

        # 1.3 Solve potential on the large coarse grid
        niter_large = self.ps_large_coar.solve(self.phi_large_coar_g,
                                               rho_large_coar_g, **kwargs)
        rho_large_coar_g = None

        if not self.use_coarse:
            deextend_array(self.gd_small_fine, self.gd_large_coar,
                           phi_small_fine_g, self.phi_large_coar_g)
            return niter_large

        if self.use_aux_grid:
            # 2.1 De-extend the potential to the auxiliary grid
            phi_aux_coar_g = self.gd_aux_coar.empty()
            deextend_array(self.gd_aux_coar, self.gd_large_coar_from_aux,
                           phi_aux_coar_g, self.phi_large_coar_g)
        else:
            phi_aux_coar_g = self.phi_large_coar_g

        # 3.1 Refine the potential
        tmp_g = phi_aux_coar_g
        for refiner in self.refiner_i:
            tmp_g = refiner.apply(tmp_g)
        phi_aux_coar_g = None
        phi_aux_fine_g = tmp_g

        # 3.2 Calculate the corresponding density with Laplace
        # (the refined coarse density would not accurately match with
        # the potential)
        rho_aux_fine_g = self.gd_aux_fine.empty()
        self.laplace_aux_fine.apply(phi_aux_fine_g, rho_aux_fine_g)

        # 3.3 De-extend the potential and density to the small grid
        cor_phi_small_fine_g = self.gd_small_fine.empty()
        deextend_array(self.gd_small_fine, self.gd_aux_fine,
                       cor_phi_small_fine_g, phi_aux_fine_g)
        phi_aux_fine_g = None
        cor_rho_small_fine_g = self.gd_small_fine.empty()
        deextend_array(self.gd_small_fine, self.gd_aux_fine,
                       cor_rho_small_fine_g, rho_aux_fine_g)
        rho_aux_fine_g = None

        # 3.4 Remove the correcting density and potential
        rho_small_fine_g -= cor_rho_small_fine_g
        phi_small_fine_g -= cor_phi_small_fine_g

        # 3.5 Solve potential on the small grid
        niter_small = self.ps_small_fine.solve(phi_small_fine_g,
                                               rho_small_fine_g, **kwargs)

        # 3.6 Correct potential and density
        phi_small_fine_g += cor_phi_small_fine_g
        # rho_small_fine_g += cor_rho_small_fine_g

        return (niter_large, niter_small)

    def estimate_memory(self, mem):
        self.ps_large_coar.estimate_memory(mem.subnode('Large grid Poisson'))
        if self.use_coarse:
            ps = self.ps_small_fine
            ps.estimate_memory(mem.subnode('Small grid Poisson'))
        mem.subnode('Large coarse phi', self.gd_large_coar.bytecount())
        tmp = max(self.gd_small_fine.bytecount(),
                  self.gd_large_coar.bytecount())
        if self.use_coarse:
            tmp = max(tmp,
                      self.gd_aux_coar.bytecount(),
                      self.gd_aux_fine.bytecount() * 2 +
                      self.gd_small_fine.bytecount(),
                      self.gd_aux_fine.bytecount() +
                      self.gd_small_fine.bytecount() * 2)
        mem.subnode('Temporary arrays', tmp)

    def get_description(self):
        line = '%s with ' % self.__class__.__name__
        if self.use_coarse:
            line += 'large and small grids'
        else:
            line += 'large grid'
        lines = [line]

        def add_line(line, pad=0):
            lines.extend(['%s%s' % (' ' * pad, line)])

        def get_cell(ps):
            descr = ps.get_description().replace('\n', '\n%s' % (' ' * 8))
            add_line('Poisson solver: %s' % descr, 8)
            if hasattr(ps, 'gd'):
                gd = ps.gd
                par = cell_to_cellpar(gd.cell_cv * Bohr)
                h_eff = gd.dv**(1.0 / 3.0) * Bohr
                l1 = '{:8d} x {:8d} x {:8d} points'.format(*gd.N_c)
                l2 = '{:8.2f} x {:8.2f} x {:8.2f} AA'.format(*par[:3])
                l3 = 'Effective grid spacing dv^(1/3) = {:.4f}'.format(h_eff)
                add_line('Grid: %s' % l1, 8)
                add_line('      %s' % l2, 8)
                add_line('      %s' % l3, 8)

        add_line('Large grid:', 4)
        get_cell(self.ps_large_coar)

        if self.use_coarse:
            add_line('Small grid:', 4)
            get_cell(self.ps_small_fine)

        return '\n'.join(lines)

    def todict(self):
        d = {'name': self.__class__.__name__}
        d['gpts'] = self.N_large_fine_c
        d['coarses'] = self.Ncoar
        d['nn_coarse'] = self.nn_coarse
        d['nn_refine'] = self.nn_refine
        d['nn_laplace'] = self.nn_laplace
        d['use_aux_grid'] = self.use_aux_grid
        d['poissonsolver_large'] = self.ps_large_coar.todict()
        if self.use_coarse:
            d['poissonsolver_small'] = self.ps_small_fine.todict()
        else:
            d['poissonsolver_small'] = None
        return d
