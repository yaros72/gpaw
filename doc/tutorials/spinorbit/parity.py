from __future__ import print_function
import numpy as np
from gpaw import GPAW
from gpaw.spinorbit import get_parity_eigenvalues

ik = 0

calc = GPAW('high_sym.gpw', txt=None)
Nv = int(calc.get_number_of_electrons() / 2)
r_v = [0, 0, 0]
multiplicities = np.array([1, 3, 3, 1])

print()
print('Calculating parities without spin-orbit coupling')
P_i = []
for ik, X in zip(range(4), ['G', 'L', 'F', 'Z']):
    ps = get_parity_eigenvalues(calc, ik=ik, spin_orbit=False, bands=range(Nv),
                                inversion_center=r_v)
    P = np.prod(ps[:Nv])
    P_i.append(P)
    print('    Parity at %s:' % X, P)
print()
print('Calculating parities with spin-orbit coupling')
Pso_i = []
for ik, X in zip(range(4), ['G', 'L', 'F', 'Z']):
    ps = get_parity_eigenvalues(calc, ik=ik, spin_orbit=True,
                                bands=range(Nv + 10),
                                inversion_center=r_v, deg_tol=1.0e-8)
    P = np.prod(ps[:Nv])
    Pso_i.append(P)
    print('    Parity at %s:' % X, P)
print()
print('Z2 index no spinorbit:',
      (1 - np.prod(np.array(P_i)**multiplicities)) / 2)
print('Z2 index spinorbit',
      (1 - np.prod(np.array(Pso_i)**multiplicities)) / 2)
