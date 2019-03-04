import numpy as np
from ase import Atoms
from ase.units import Bohr
from gpaw.jellium import Jellium
from gpaw import GPAW, PW

rs = 5.0 * Bohr  # Wigner-Seitz radius
h = 0.2          # grid-spacing
a = 8 * h        # lattice constant
k = 12           # number of k-points (k*k*k)

ne = a**3 / (4 * np.pi / 3 * rs**3)
jellium = Jellium(ne)

bulk = Atoms(pbc=True, cell=(a, a, a))
bulk.calc = GPAW(mode=PW(400.0),
                 background_charge=jellium,
                 xc='LDA_X+LDA_C_WIGNER',
                 nbands=5,
                 kpts=[k, k, k],
                 h=h,
                 txt='bulk.txt')
e0 = bulk.get_potential_energy()
