import numpy as np
from ase.build import bulk
from ase.optimize.bfgs import BFGS
from ase.constraints import UnitCellFilter
from gpaw import GPAW
from gpaw import PW

si = bulk('Si', 'fcc', a=6.0)
# Experimental Lattice constant is a=5.421 A

si.calc = GPAW(xc='PBE',
               mode=PW(400, dedecut='estimate'),
               kpts=(4, 4, 4),
               # convergence={'eigenstates': 1.e-10},  # converge tightly!
               txt='stress.txt')

uf = UnitCellFilter(si)
relax = BFGS(uf)
relax.run(fmax=0.05)  # Consider much tighter fmax!

a = np.linalg.norm(si.cell[0]) * 2**0.5
print('Relaxed lattice parameter: a = {} Ang'.format(a))
