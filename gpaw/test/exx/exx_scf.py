"""Test selfconsistent RSF calculation with Yukawa potential including vc."""
from ase import Atoms
from gpaw import GPAW, setup_paths, KohnShamConvergenceError
from gpaw.xc.hybrid import HybridXC
from gpaw.poisson import PoissonSolver
from gpaw.occupations import FermiDirac
from gpaw.test import gen
from gpaw.eigensolvers import RMMDIIS
from gpaw.cluster import Cluster

if setup_paths[0] != '.':
    setup_paths.insert(0, '.')

h = 0.3

# No energies - simpley convergence test, esp. for 3d TM

# for atom in ['F', 'Cl', 'Br', 'Cu', 'Ag']:
for atom in ['Ti']:
    gen(atom, xcname='PBE', scalarrel=False, exx=True)
    work_atom = Cluster(Atoms(atom, [(0, 0, 0)]))
    work_atom.minimal_box(4, h=h)
    work_atom.translate([0.01, 0.02, 0.03])
    work_atom.set_initial_magnetic_moments([2.0])
    calculator = GPAW(convergence={'energy': 0.01,
                                   'eigenstates': 3,
                                   'density': 3},
                      eigensolver=RMMDIIS(),
                      poissonsolver=PoissonSolver(use_charge_center=True),
                      occupations=FermiDirac(width=0.0, fixmagmom=True),
                      h=h, maxiter=35)   # Up to 24 are needed by now
    calculator.set(xc=HybridXC('PBE0'))
    calculator.set(txt=atom + '-PBE0.txt')
    work_atom.set_calculator(calculator)
    try:
        work_atom.get_potential_energy()
    except KohnShamConvergenceError:
        pass
    assert calculator.scf.converged, 'Calculation not converged'
