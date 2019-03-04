from gpaw import GPAW
from ase.build import fcc111
from ase.build import add_adsorbate
atoms = fcc111('Pt', (2, 2, 6), a=4.00, vacuum=10.0)
add_adsorbate(atoms, 'O', 2.0, 'fcc')
calc = GPAW(mode='pw',
            kpts=(8, 8, 1),
            xc='RPBE')
ncpus = 4
