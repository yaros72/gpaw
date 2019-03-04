import os
import gpaw.wannier90 as w90
from gpaw import GPAW

seed = 'Cu'

calc = GPAW(seed + '.gpw', txt=None)

w90.write_input(calc,
                orbitals_ai=[[0, 1, 4, 5, 6, 7, 8]],
                bands=range(20),
                num_iter=1000,
                dis_num_iter=500)

os.system('wannier90.x -pp ' + seed)

w90.write_projections(calc, orbitals_ai=[[0, 1, 4, 5, 6, 7, 8]])
w90.write_eigenvalues(calc)
w90.write_overlaps(calc)

os.system('wannier90.x ' + seed)
