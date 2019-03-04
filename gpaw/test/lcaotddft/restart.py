import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.poisson import PoissonSolver
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.mpi import world

from gpaw.test import equal

# Atoms
atoms = molecule('SiH4')
atoms.center(vacuum=4.0)

# Ground-state calculation
calc = GPAW(nbands=7, h=0.4,
            basis='dzp', mode='lcao',
            poissonsolver=PoissonSolver(eps=1e-16),
            convergence={'density': 1e-8},
            xc='GLLBSC',
            txt='gs.out')
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')

# Time-propagation calculation
td_calc = LCAOTDDFT('gs.gpw', txt='td.out')
DipoleMomentWriter(td_calc, 'dm.dat')
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 3)

# Write a restart point
td_calc.write('td.gpw', mode='all')

# Keep propagating
td_calc.propagate(20, 3)

# Restart from the restart point
td_calc = LCAOTDDFT('td.gpw', txt='td2.out')
DipoleMomentWriter(td_calc, 'dm.dat')
td_calc.propagate(20, 3)
world.barrier()

# Check dipole moment file
data_tj = np.loadtxt('dm.dat')
# Original run
ref_i = data_tj[4:8].ravel()
# Restarted steps
data_i = data_tj[8:].ravel()

tol = 1e-10
equal(data_i, ref_i, tol)

# Test the absolute values
data = np.loadtxt('dm.dat')[:8].ravel()
if 0:
    from gpaw.test import print_reference
    print_reference(data, 'ref', '%.12le')

ref = [0.0, 2.16436889e-15, -3.332079476367e-14, -2.770920572229e-14,
       -2.124228060013e-14, 0.0, -5.50792334e-16, -1.480925618326e-14,
       2.865764301706e-15, 1.117595510483e-15, 0.82682747, -3.84328149e-16,
       6.205228222006e-05, 6.205228223474e-05, 6.205228223109e-05,
       1.65365493, -4.4704896e-16, 0.0001001904123067, 0.0001001904123111,
       0.0001001904123113, 2.4804824, -2.73032778e-15, 0.0001069907981589,
       0.0001069907981663, 0.0001069907981655, 3.30730987, 3.13998397e-16,
       9.1907609201e-05, 9.190760920932e-05, 9.190760920682e-05, 4.13413733,
       -1.52984486e-15, 6.808339313731e-05, 6.808339313977e-05,
       6.80833931417e-05, 4.9609648, -2.74286225e-15, 4.135685385213e-05,
       4.135685383555e-05, 4.135685383795e-05]

print('result')
print(data.tolist())

tol = 1e-12
equal(data, ref, tol)
