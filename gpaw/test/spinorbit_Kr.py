from ase import Atoms
from gpaw import GPAW
from gpaw.spinorbit import get_spinorbit_eigenvalues
from gpaw.test import equal

a = Atoms('Kr')
a.center(vacuum=3.0)

calc = GPAW(mode='pw', xc='LDA')

a.calc = calc
a.get_potential_energy()

e_n = calc.get_eigenvalues()
e_m = get_spinorbit_eigenvalues(calc)

equal(e_n[0] - e_m[0], 0.0, 1.0e-3)
equal(e_n[1] - e_m[2], 0.452, 1.0e-3)
equal(e_n[2] - e_m[4], -0.226, 1.0e-3)
