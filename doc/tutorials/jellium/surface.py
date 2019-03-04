import numpy as np
from ase import Atoms
from ase.units import Bohr
from gpaw.jellium import JelliumSlab
from gpaw import GPAW, PW

rs = 5.0 * Bohr  # Wigner-Seitz radius
h = 0.2          # grid-spacing
a = 8 * h        # lattice constant
v = 3 * a        # vacuum
L = 10 * a       # thickness
k = 12           # number of k-points (k*k*1)

ne = a**2 * L / (4 * np.pi / 3 * rs**3)

jellium = JelliumSlab(ne, z1=v, z2=v + L)

surf = Atoms(pbc=(True, True, False),
             cell=(a, a, v + L + v))
surf.calc = GPAW(mode=PW(400.0),
                 background_charge=jellium,
                 xc='LDA_X+LDA_C_WIGNER',
                 eigensolver='dav',
                 kpts=[k, k, 1],
                 h=h,
                 convergence={'density': 0.001},
                 nbands=int(ne / 2) + 15,
                 txt='surface.txt')
e = surf.get_potential_energy()
surf.calc.write('surface.gpw')
