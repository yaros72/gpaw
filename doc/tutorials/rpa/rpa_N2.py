from __future__ import print_function
from ase.parallel import paropen
from ase.units import Hartree
from gpaw.xc.rpa import RPACorrelation

rpa = RPACorrelation('N.gpw', nblocks=8, truncation='wigner-seitz',
                     txt='rpa_N.txt')
E1_i = rpa.calculate(ecut=400)

rpa = RPACorrelation('N2.gpw', nblocks=8, truncation='wigner-seitz',
                     txt='rpa_N2.txt')
E2_i = rpa.calculate(ecut=400)
ecut_i = rpa.ecut_i

f = paropen('rpa_N2.dat', 'w')
for ecut, E1, E2 in zip(ecut_i, E1_i, E2_i):
    print(ecut * Hartree, E2 - 2 * E1, file=f)
f.close()
