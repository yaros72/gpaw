# Creates: qsfdtd_vs_mie.png
import numpy as np
import pylab as plt
from ase.units import Hartree, Bohr
from gpaw.fdtd.polarizable_material import PermittivityPlus, _eps0_au

# Nanosphere radius (Angstroms)
radius = 50.0

# Permittivity of Gold
# J. Chem. Phys. 135, 084121 (2011); http://dx.doi.org/10.1063/1.3626549
gold = [[0.2350, 0.1551, 95.62],
        [0.4411, 0.1480, -12.55],
        [0.7603, 1.946, -40.89],
        [1.161, 1.396, 17.22],
        [2.946, 1.183, 15.76],
        [4.161, 1.964, 36.63],
        [5.747, 1.958, 22.55],
        [7.912, 1.361, 81.04]]

# Plot calculated spectrum and compare with Mie theory
spec = np.loadtxt('spec.dat')
perm = PermittivityPlus(data=gold).value(spec[:, 0] / Hartree)
plt.figure()
plt.plot(spec[:, 0], spec[:, 1], 'r', label='QSFDTD')
plt.plot(spec[:, 0],
         3. * (4. / 3. * np.pi * (radius / Bohr)**3) *
         (spec[:, 0] / Hartree) / (2. * np.pi**2) / Hartree *
         np.imag((perm - _eps0_au) / (perm + 2. * _eps0_au)),
         'b', label='Mie theory')
plt.legend(loc=2)
plt.xlabel('Energy [eV]', fontsize=12)
plt.ylabel('Dipole strength [1/eV]', fontsize=12)
plt.xlim((0, 5.0))
plt.ylim((-1, 3500))
plt.savefig('qsfdtd_vs_mie.png')
