from gpaw.solvation.poisson import (WeightedFDPoissonSolver,
                                    ADM12PoissonSolver,
                                    PolarizationPoissonSolver)
from gpaw.solvation.dielectric import Dielectric
from gpaw.grid_descriptor import GridDescriptor
from gpaw.utilities.gauss import Gaussian
from gpaw.fd_operators import Gradient
from ase.units import Bohr
from gpaw.test import equal
from gpaw.utilities import erf
from ase.parallel import parprint
import numpy as np
import warnings

nn = 3
accuracy = 2e-10


def gradient(gd, x, nn):
    out = gd.empty(3)
    for i in (0, 1, 2):
        Gradient(gd, i, 1.0, nn).apply(x, out[i])
    return out


def make_gd(h, box, pbc):
    diag = np.array([box] * 3)
    cell = np.diag(diag)
    grid_shape = tuple((diag / h * 2).astype(int))
    return GridDescriptor(grid_shape, cell / Bohr, pbc)


class MockDielectric(Dielectric):
    def __init__(self, epsinf, nn):
        Dielectric.__init__(self, epsinf)
        self.nn = nn

    def update(self, cavity):
        grad_eps_vg = gradient(self.gd, self.eps_gradeps[0] - self.epsinf,
                               self.nn)
        for i in (0, 1, 2):
            self.eps_gradeps[1 + i][...] = grad_eps_vg[i]


box = 12.
gd = make_gd(h=.4, box=box, pbc=False)


def solve(ps, eps, rho):
    phi = gd.zeros()
    dielectric = MockDielectric(epsinf=eps.max(), nn=nn)
    dielectric.set_grid_descriptor(gd)
    dielectric.allocate()
    dielectric.eps_gradeps[0][...] = eps
    dielectric.update(None)
    with warnings.catch_warnings():
        # ignore production code warning for alternative psolvers
        warnings.simplefilter("ignore")
        solver = ps(nn=nn, relax='J', eps=accuracy)
    solver.set_dielectric(dielectric)
    solver.set_grid_descriptor(gd)
    solver.solve(phi, rho)
    return phi


psolvers = (WeightedFDPoissonSolver,
            ADM12PoissonSolver,
            PolarizationPoissonSolver)


# test neutral system with constant permittivity
parprint('neutral, constant permittivity')
epsinf = 80.
eps = gd.zeros()
eps.fill(epsinf)
qs = (-1., 1.)
shifts = (-1., 1.)
rho = gd.zeros()
phi_expected = gd.zeros()
for q, shift in zip(qs, shifts):
    gauss_norm = q / np.sqrt(4 * np.pi)
    gauss = Gaussian(gd, center=(box / 2. + shift) * np.ones(3) / Bohr)
    rho += gauss_norm * gauss.get_gauss(0)
    phi_expected += gauss_norm * gauss.get_gauss_pot(0) / epsinf

for ps in psolvers:
    phi = solve(ps, eps, rho)
    parprint(ps, np.abs(phi - phi_expected).max())
    equal(phi, phi_expected, 1e-3)


# test charged system with constant permittivity
parprint('charged, constant permittivity')
epsinf = 80.
eps = gd.zeros()
eps.fill(epsinf)
q = -2.
gauss_norm = q / np.sqrt(4 * np.pi)
gauss = Gaussian(gd, center=(box / 2. + 1.) * np.ones(3) / Bohr)
rho_gauss = gauss_norm * gauss.get_gauss(0)
phi_gauss = gauss_norm * gauss.get_gauss_pot(0)
phi_expected = phi_gauss / epsinf

for ps in psolvers:
    phi = solve(ps, eps, rho_gauss)
    parprint(ps, np.abs(phi - phi_expected).max())
    equal(phi, phi_expected, 1e-3)


# test non-constant permittivity
msgs = ('neutral, non-constant permittivity',
        'charged, non-constant permittivity')
qss = ((-1., 1.), (2., ))
shiftss = ((-.4, .4), (-1., ))
epsshifts = (.0, -1.)

for msg, qs, shifts, epsshift in zip(msgs, qss, shiftss, epsshifts):
    parprint(msg)
    epsinf = 80.
    gauss = Gaussian(gd, center=(box / 2. + epsshift) * np.ones(3) / Bohr)
    eps = gauss.get_gauss(0)
    eps = epsinf - eps / eps.max() * (epsinf - 1.)

    rho = gd.zeros()
    phi_expected = gd.zeros()
    grad_eps = gradient(gd, eps - epsinf, nn)

    for q, shift in zip(qs, shifts):
        gauss = Gaussian(gd, center=(box / 2. + shift) * np.ones(3) / Bohr)
        phi_tmp = gauss.get_gauss_pot(0)
        xyz = gauss.xyz
        fac = 2. * np.sqrt(gauss.a) * np.exp(-gauss.a * gauss.r2)
        fac /= np.sqrt(np.pi) * gauss.r2
        fac -= erf(np.sqrt(gauss.a) * gauss.r) / (gauss.r2 * gauss.r)
        fac *= 2.0 * 1.7724538509055159
        grad_phi = fac * xyz
        laplace_phi = -4. * np.pi * gauss.get_gauss(0)
        rho_tmp = -1. / (4. * np.pi) * (
            (grad_eps * grad_phi).sum(0) + eps * laplace_phi)
        norm = gd.integrate(rho_tmp)
        rho_tmp /= norm * q
        phi_tmp /= norm * q
        rho += rho_tmp
        phi_expected += phi_tmp

    # PolarizationPoissonSolver does not pass this test
    for ps in psolvers[:-1]:
        phi = solve(ps, eps, rho)
        parprint(ps, np.abs(phi - phi_expected).max())
        equal(phi, phi_expected, 1e-3)
