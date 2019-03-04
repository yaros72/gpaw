"""Calculations using RSF with dedicated datasets and Poisson-solver."""
from ase import Atoms
from gpaw import GPAW, setup_paths
from gpaw.poisson import PoissonSolver
from gpaw.eigensolvers import RMMDIIS
from gpaw.occupations import FermiDirac
from gpaw.test import gen

if setup_paths[0] != '.':
    setup_paths.insert(0, '.')

for atom in ['C', 'O']:
    gen(atom, xcname='PBE', scalarrel=True, exx=True,
        yukawa_gamma=0.75)

h = 0.30
co = Atoms('CO', positions=[(0, 0, 0), (0, 0, 1.15)])
co.center(5)

# c = {'energy': 0.005, 'eigenstates': 1e-4}  # Usable values
c = {'energy': 0.1, 'eigenstates': 3, 'density': 3}  # Values for test

calc = GPAW(txt='CO.txt', xc='LCY-PBE', convergence=c,
            eigensolver=RMMDIIS(), h=h,
            poissonsolver=PoissonSolver(use_charge_center=True),
            occupations=FermiDirac(width=0.0), spinpol=False)
co.set_calculator(calc)
co.get_potential_energy()
