import numpy as np

from ase import Atoms
from ase.parallel import paropen

from gpaw import GPAW, FermiDirac, Davidson
from gpaw.wavefunctions.pw import PW
from gpaw.response.chi0 import Chi0
from gpaw.test import equal

# This script asserts that the chi's obtained
# from GS calculations using symmetries
# and GS calculations not using symmetries return
# the same results. Tests that the chi's are element-wise
# equal to a tolerance of 1e-10.

resultfile = paropen('results.txt', 'a')
pwcutoff = 400.0
k = 4
a = 4.59
c = 2.96
u = 0.305

rutile_cell = [[a, 0, 0],
               [0, a, 0],
               [0, 0, c]]

TiO2_basis = np.array([[0.0, 0.0, 0.0],
                       [0.5, 0.5, 0.5],
                       [u, u, 0.0],
                       [-u, -u, 0.0],
                       [0.5 + u, 0.5 - u, 0.5],
                       [0.5 - u, 0.5 + u, 0.5]])

bulk_crystal = Atoms(symbols=['Ti', 'Ti', 'O', 'O', 'O', 'O'],
                     scaled_positions=TiO2_basis,
                     cell=rutile_cell,
                     pbc=(1, 1, 1))
data_s = []
for symmetry in ['off', {}]:
    bulk_calc = GPAW(mode=PW(pwcutoff),
                     nbands=42,
                     eigensolver=Davidson(1),
                     kpts={'size': (k, k, k), 'gamma': True},
                     xc='PBE',
                     occupations=FermiDirac(0.00001),
                     parallel={'band': 1},
                     symmetry=symmetry)

    bulk_crystal.set_calculator(bulk_calc)
    e0_bulk_pbe = bulk_crystal.get_potential_energy()
    bulk_calc.write('bulk.gpw', mode='all')
    X = Chi0('bulk.gpw')
    chi_t = X.calculate([1. / 4, 0, 0])[1:]
    data_s.append(list(chi_t))

msg = 'Difference in Chi when turning off symmetries!'

while len(data_s):
    data1 = data_s.pop()
    for data2 in data_s:
        for dat1, dat2 in zip(data1, data2):
            if dat1 is not None:
                equal(np.abs(dat1 - dat2).max(),
                      0, 1e-5, msg=msg)
