from __future__ import print_function
from ase.dft.kpoints import ibz_points, bandpath
from ase.parallel import paropen
from gpaw import GPAW

layer = GPAW('Fe_gs.gpw', txt=None).atoms

points = ibz_points['bcc']
G = points['Gamma']
H = points['H']
P = points['P']
N = points['N']
H_z = [H[0], -H[1], -H[2]]
G_yz = [2 * H[0], 0.0, 0.0]

kpts, x, X = bandpath([G, H, G_yz], layer.cell, npoints=1000)
calc = GPAW('Fe_gs.gpw',
            kpts=kpts,
            fixdensity=True,
            symmetry='off',
            txt='Fe_bands.txt',
            parallel={'band': 1})
calc.get_potential_energy()

calc.write('Fe_bands.gpw')

f = paropen('Fe_kpath.dat', 'w')
for k in x:
    print(k, file=f)
f.close()

f = paropen('Fe_highsym.dat', 'w')
for k in X:
    print(k, file=f)
f.close()
