from ase.build import molecule

from gpaw.test import equal
import gpaw.solvation as solv
from gpaw.cluster import Cluster
from gpaw.lrtddft import LrTDDFT
from gpaw import PoissonSolver

h = 0.3
vac = 3.0

atoms = Cluster(molecule('H2'))
atoms.minimal_box(vac, h)

# convenient way to use HW14 water parameters:
calc = solv.SolvationGPAW(
    xc='PBE', h=0.2,  # non-solvent DFT parameters
    nbands=3,
    convergence={'energy': 0.1, 'eigenstates': 0.01, 'density': 0.1},
    **solv.get_HW14_water_kwargs())

atoms.set_calculator(calc)
atoms.get_potential_energy()
print(id(calc.hamiltonian.poisson.dielectric))
print(id(calc.hamiltonian.dielectric))


# We test the agreement of a pure RPA kernel with setting eps to 1
lr = LrTDDFT(calc, poisson=PoissonSolver('fd', nn=calc.hamiltonian.poisson.nn))
lr.diagonalize()

if 0:
    lreps = LrTDDFT(calc)
    lreps.diagonalize()

calc.hamiltonian.poisson.dielectric.epsinf = 1.
lr1 = LrTDDFT(calc)
lr1.diagonalize()
for ex, ex1 in zip(lr, lr1):
    equal(ex.energy, ex1.energy, 1e-14)
