import os
import gpaw.wannier90 as w90
from gpaw import GPAW

seed = 'GaAs'

calc = GPAW(seed + '.gpw', txt=None)

w90.write_input(calc, orbitals_ai=[[], [0, 1, 2, 3]],
                bands=range(4),
                seed=seed,
                num_iter=1000,
                plot=True)
w90.write_wavefunctions(calc, seed=seed)
os.system('wannier90.x -pp ' + seed)

w90.write_projections(calc, orbitals_ai=[[], [0, 1, 2, 3]], seed=seed)
w90.write_eigenvalues(calc, seed=seed)
w90.write_overlaps(calc, seed=seed)

os.system('wannier90.x ' + seed)
