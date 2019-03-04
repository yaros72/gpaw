from __future__ import print_function
from ase.dft.kpoints import bandpath
from ase.parallel import paropen
from gpaw import GPAW

layer = GPAW('WS2_gs.gpw', txt=None).atoms

G = [0, 0, 0]
K = [1 / 3., 1 / 3., 0]
M = [0.5, 0, 0]
M_ = [-0.5, 0, 0]
K_ = [-1 / 3., -1 / 3., 0]
kpts, x, X = bandpath([M, K, G, K_, M_], layer.cell, npoints=1000)

calc = GPAW('WS2_gs.gpw', kpts=kpts, symmetry='off', fixdensity=True)
calc.get_potential_energy()

calc.write('WS2_bands.gpw')

f = paropen('WS2_kpath.dat', 'w')
for k in x:
    print(k, file=f)
f.close()

f = paropen('WS2_highsym.dat', 'w')
for k in X:
    print(k, file=f)
f.close()
