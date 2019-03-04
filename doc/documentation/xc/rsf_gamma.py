"""Calculation utilizing RSF with optimized gamma."""
from ase import Atoms
from gpaw import GPAW, setup_paths
from gpaw.poisson import PoissonSolver
from gpaw.eigensolvers import RMMDIIS
from gpaw.occupations import FermiDirac
from gpaw.test import equal, gen

# IP for CO using LCY-PBE with gamma=0.81 after
# dx.doi.org/10.1021/acs.jctc.8b00238
IP = 14.31

if setup_paths[0] != '.':
    setup_paths.insert(0, '.')

for atom in ['C', 'O']:
    gen(atom, xcname='PBE', scalarrel=True, exx=True,
        yukawa_gamma=0.81)

h = 0.30
co = Atoms('CO', positions=[(0, 0, 0), (0, 0, 1.15)])
co.minimal_box(5)

# c = {'energy': 0.005, 'eigenstates': 1e-4}  # Usable values
c = {'energy': 0.1, 'eigenstates': 3, 'density': 3}  # Values for test

calc = GPAW(txt='CO.txt', xc='LCY-PBE:omega=0.81', convergence=c,
            eigensolver=RMMDIIS(), h=h,
            poissonsolver=PoissonSolver(use_charge_center=True),
            occupations=FermiDirac(width=0.0), spinpol=False)
co.set_calculator(calc)
co.get_potential_energy()
(eps_homo, eps_lumo) = calc.get_homo_lumo()
equal(eps_homo, -IP, 0.15)


