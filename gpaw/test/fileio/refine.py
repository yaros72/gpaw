"""Test automatically write out of restart files"""
from __future__ import print_function

from ase import Atoms
from gpaw import GPAW
from gpaw.test import equal

restart_wf = 'gpaw-restart-wf.gpw'
# H2
H = Atoms('HH', [(0, 0, 0), (0, 0, 1)])
H.center(vacuum=2.0)

if 1:
    calc = GPAW(nbands=2,
                convergence={'eigenstates': 0.001,
                             'energy': 0.1,
                             'density': 0.1})
    H.set_calculator(calc)
    H.get_potential_energy()
    calc.write(restart_wf, 'all')

    # refine the result directly
    calc.set(convergence={'energy': 0.00001})
    Edirect = H.get_potential_energy()

# refine the result after reading from a file
H = GPAW(restart_wf, convergence={'energy': 0.00001}).get_atoms()
Erestart = H.get_potential_energy()

print(Edirect, Erestart)
# Note: the different density mixing introduces small differences
equal(Edirect, Erestart, 4e-5)
