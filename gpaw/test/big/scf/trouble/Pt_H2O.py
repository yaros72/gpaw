from ase.io import read
from gpaw import GPAW

atoms = read('Pt_H2O.xyz')
atoms.set_cell([[8.527708, 0, 0],
               [0, 4.923474, 0],
               [0, 0, 16]],
               scale_atoms=False)
atoms.center(axis=2)

atoms.pbc = (True, True, False)

atoms.calc = GPAW(h=0.20,
                  kpts=(2, 4, 1),
                  xc='RPBE',
                  poissonsolver={'dipolelayer': 'xy'},
                  basis='dzp',
                  maxiter=200,
                  txt='Pt_H2O.txt')
ncpus = 8
