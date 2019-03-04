"""Self-consistent SOC."""
from unittest import SkipTest

import numpy as np
from ase.build import mx2

from gpaw import GPAW
from gpaw.test import equal
from gpaw.spinorbit import get_spinorbit_eigenvalues
from gpaw.mpi import size


if size > 1:
    raise SkipTest()


a = mx2('MoS2')
a.center(vacuum=3, axis=2)

params = dict(mode='pw',
              kpts={'size': (3, 3, 1),
                    'gamma': True})

# Selfconsistent:
a.calc = GPAW(experimental={'magmoms': np.zeros((3, 3)),
                            'soc': True},
              convergence={'bands': 28},
              **params)
a.get_potential_energy()
E1 = a.calc.get_eigenvalues(kpt=2)

# Non-selfconsistent:
a.calc = GPAW(convergence={'bands': 14}, **params)
a.get_potential_energy()
E2 = get_spinorbit_eigenvalues(a.calc, bands=np.arange(14))[:, 2]


def test(E, hsplit, lsplit):
    print(E)
    h1, h2, l1, l2 = E[24:28]  # HOMO-1, HOMO, LUMO, LUMP+1
    print(h2 - h1)
    print(l2 - l1)
    assert abs(h2 - h1 - hsplit) < 0.01
    assert abs(l2 - l1 - lsplit) < 0.002


equal(E1[:28], E2, tolerance=1e-1)
test(E1, 0.15, 0.002)
test(E2, 0.15, 0.007)
