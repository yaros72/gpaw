import numpy as np
from ase.units import Bohr
from gpaw.poisson import FDPoissonSolver
from gpaw.utilities.gauss import Gaussian
from gpaw.utilities.timing import nulltimer

from ase.utils.timing import timer

from gpaw.utilities.extend_grid import extended_grid_descriptor, \
    extend_array, deextend_array


class ExtendedPoissonSolver(FDPoissonSolver):
    """ExtendedPoissonSolver

    Parameter syntax:

    moment_corrections = [{'moms': moms_list1, 'center': center1},
                          {'moms': moms_list2, 'center': center2},
                          ...]
    Here moms_listX is list of integers of multipole moments to be corrected
    at centerX.

    extended = {'gpts': gpts, 'useprev': useprev}
    Here gpts is number of grid points in the **coarse** grid corresponding
    to the larger grid used for PoissonSolver (the Poisson equation
    is solved on fine grid as usual, but the gpts is given in coarse grid
    units for convenience), and useprev is boolean determining whether previous
    solution of the PoissonSolver instance is used as an initial guess
    for the next solve() call.

    Important: provide timer for PoissonSolver to analyze the cost of
    the multipole moment corrections and grid extension to your system!

    """
    # TODO: enable 'comm' parameter for 'extended' dictionary. This would
    # allow to use the whole mpi.world for PoissonSolver.
    # Currently, Poisson equation calculation is duplicated over, e.g.,
    # band and kpt communicators.

    def __init__(self, nn=3, relax='J', eps=2e-10, maxiter=1000,
                 moment_corrections=None,
                 extended=None,
                 timer=nulltimer):

        FDPoissonSolver.__init__(self, nn=nn, relax=relax,
                                 eps=eps, maxiter=maxiter,
                                 remove_moment=None)

        self.timer = timer

        if moment_corrections is None:
            self.moment_corrections = None
        elif isinstance(moment_corrections, int):
            self.moment_corrections = [{'moms': range(moment_corrections),
                                        'center': None}]
        else:
            self.moment_corrections = moment_corrections

        self.is_extended = False
        # Broadcast over band, kpt, etc. communicators required?
        self.requires_broadcast = False
        if extended is not None:
            self.is_extended = True
            self.extended = extended
            assert 'gpts' in extended.keys(), 'gpts parameter is missing'
            self.extended['gpts'] = np.array(self.extended['gpts'])
            # Multiply gpts by 2 to get gpts on fine grid
            self.extended['finegpts'] = self.extended['gpts'] * 2
            assert 'useprev' in extended.keys(), 'useprev parameter is missing'
            if self.extended.get('comm') is not None:
                self.requires_broadcast = True

    def set_grid_descriptor(self, gd):
        if self.is_extended:
            self.gd_original = gd
            assert np.all(self.gd_original.N_c <= self.extended['finegpts']), \
                'extended grid has to be larger than the original one'
            gd, _, _ = extended_grid_descriptor(
                gd,
                N_c=self.extended['finegpts'],
                extcomm=self.extended.get('comm'))
        FDPoissonSolver.set_grid_descriptor(self, gd)

    def get_description(self):
        description = FDPoissonSolver.get_description(self)

        lines = [description]

        if self.is_extended:
            lines.extend(['    Extended %d*%d*%d grid' %
                          tuple(self.gd.N_c)])
            lines.extend(['    Use previous is %s' %
                          self.extended['useprev']])

        if self.moment_corrections:
            lines.extend(['    %d moment corrections:' %
                          len(self.moment_corrections)])
            lines.extend(['      %s' %
                          ('[%s] %s' %
                           ('center' if mom['center'] is None
                            else (', '.join(['%.2f' % (x * Bohr)
                                             for x in mom['center']])),
                            mom['moms']))
                          for mom in self.moment_corrections])

        return '\n'.join(lines)

    @timer('Poisson initialize')
    def _init(self):
        if self._initialized:
            return
        FDPoissonSolver._init(self)

        if self.is_extended:
            if not self.gd.orthogonal or self.gd.pbc_c.any():
                raise NotImplementedError('Only orthogonal unit cells '
                                          'and non-periodic boundary '
                                          'conditions are tested')
            self.rho_g = self.gd.zeros()
            self.phi_g = self.gd.zeros()

        if self.moment_corrections is not None:
            if not self.gd.orthogonal or self.gd.pbc_c.any():
                raise NotImplementedError('Only orthogonal unit cells '
                                          'and non-periodic boundary '
                                          'conditions are tested')
            self.load_moment_corrections_gauss()

    @timer('Load moment corrections')
    def load_moment_corrections_gauss(self):
        if not hasattr(self, 'gauss_i'):
            self.gauss_i = []
            mask_ir = []
            r_ir = []
            self.mom_ij = []

            for rmom in self.moment_corrections:
                if rmom['center'] is None:
                    center = None
                else:
                    center = np.array(rmom['center'])
                mom_j = rmom['moms']
                gauss = Gaussian(self.gd, center=center)
                self.gauss_i.append(gauss)
                r_ir.append(gauss.r.ravel())
                mask_ir.append(self.gd.zeros(dtype=int).ravel())
                self.mom_ij.append(mom_j)

            r_ir = np.array(r_ir)
            mask_ir = np.array(mask_ir)

            Ni = r_ir.shape[0]
            Nr = r_ir.shape[1]

            for r in range(Nr):
                i = np.argmin(r_ir[:, r])
                mask_ir[i, r] = 1

            self.mask_ig = []
            for i in range(Ni):
                mask_r = mask_ir[i]
                mask_g = mask_r.reshape(self.gd.n_c)
                self.mask_ig.append(mask_g)

                # Uncomment this to see masks on grid
                # big_g = self.gd.collect(mask_g)
                # if self.gd.comm.rank == 0:
                #     big_g.dump('mask_%dg' % (i))

    def solve(self, phi, rho, charge=None, eps=None, maxcharge=1e-6,
              zero_initial_phi=False):
        self._init()
        if self.is_extended:
            self.rho_g[:] = 0
            if not self.extended['useprev']:
                self.phi_g[:] = 0

            self.timer.start('Extend array')
            extend_array(self.gd_original, self.gd, rho, self.rho_g)
            self.timer.stop('Extend array')

            retval = self._solve(self.phi_g, self.rho_g, charge,
                                 eps, maxcharge, zero_initial_phi)

            self.timer.start('Deextend array')
            deextend_array(self.gd_original, self.gd, phi, self.phi_g)
            self.timer.stop('Deextend array')

            return retval
        else:
            return self._solve(phi, rho, charge, eps, maxcharge,
                               zero_initial_phi)

    @timer('Solve')
    def _solve(self, phi, rho, charge=None, eps=None, maxcharge=1e-6,
               zero_initial_phi=False):
        self._init()
        if eps is None:
            eps = self.eps

        if self.moment_corrections:
            assert not self.gd.pbc_c.any()

            self.timer.start('Multipole moment corrections')

            rho_neutral = rho * 0.0
            phi_cor_k = []
            for gauss, mask_g, mom_j in zip(self.gauss_i, self.mask_ig,
                                            self.mom_ij):
                rho_masked = rho * mask_g
                for mom in mom_j:
                    phi_cor_k.append(gauss.remove_moment(rho_masked, mom))
                rho_neutral += rho_masked

            # Remove multipoles for better initial guess
            for phi_cor in phi_cor_k:
                phi -= phi_cor

            self.timer.stop('Multipole moment corrections')

            self.timer.start('Solve neutral')
            niter = self.solve_neutral(phi, rho_neutral, eps=eps)
            self.timer.stop('Solve neutral')

            self.timer.start('Multipole moment corrections')
            # correct error introduced by removing multipoles
            for phi_cor in phi_cor_k:
                phi += phi_cor
            self.timer.stop('Multipole moment corrections')

            return niter
        else:
            return FDPoissonSolver.solve(self, phi, rho, charge,
                                         eps, maxcharge,
                                         zero_initial_phi)

    def estimate_memory(self, mem):
        FDPoissonSolver.estimate_memory(self, mem)
        gdbytes = self.gd.bytecount()
        if self.is_extended:
            mem.subnode('extended arrays',
                        2 * gdbytes)
        if self.moment_corrections is not None:
            mem.subnode('moment_corrections masks',
                        len(self.moment_corrections) * gdbytes)

    def __repr__(self):
        template = 'ExtendedPoissonSolver(relax=\'%s\', nn=%s, eps=%e)'
        representation = template % (self.relax, repr(self.nn), self.eps)
        return representation
