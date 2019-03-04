from ase import Atoms
from ase.units import Hartree, Bohr

from gpaw import GPAW
from gpaw.xc import XC
from gpaw.test import equal
from gpaw.xc.kernel import XCNull
from gpaw.poisson import NoInteractionPoissonSolver
from gpaw.external import ExternalPotential


a = 4.0
x = Atoms(cell=(a, a, a))  # no atoms


class HarmonicPotential(ExternalPotential):
    def calculate_potential(self, gd):
        r_vg = gd.get_grid_point_coordinates()
        self.vext_g = 0.5 * ((r_vg - a / Bohr / 2)**2).sum(0)

    def get_name(self):
        return 'HarmonicPotential'


calc = GPAW(charge=-8,
            nbands=4,
            h=0.2,
            xc=XC(XCNull()),
            external=HarmonicPotential(),
            poissonsolver=NoInteractionPoissonSolver(),
            eigensolver='cg')

x.calc = calc
x.get_potential_energy()

eigs = calc.get_eigenvalues()
equal(eigs[0], 1.5 * Hartree, 0.002)
equal(abs(eigs[1:] - 2.5 * Hartree).max(), 0, 0.003)
