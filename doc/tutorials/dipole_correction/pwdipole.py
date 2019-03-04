from ase.build import fcc100, add_adsorbate
from gpaw import GPAW, PW

slab = fcc100('Al', (2, 2, 2), a=4.05, vacuum=7.5)
add_adsorbate(slab, 'Na', 4.0)
slab.center(axis=2)

slab.calc = GPAW(mode=PW(300),
                 poissonsolver={'dipolelayer': 'xy'},
                 txt='pwcorrected.txt',
                 xc='PBE',
                 setups={'Na': '1'},
                 kpts=(4, 4, 1))
e3 = slab.get_potential_energy()
slab.calc.write('pwcorrected.gpw')
