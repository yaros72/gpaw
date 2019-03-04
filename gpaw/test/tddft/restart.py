import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.tddft import TDDFT
from gpaw.poisson import PoissonSolver
from gpaw.mpi import world

from gpaw.test import equal

# Atoms
atoms = molecule('SiH4')
atoms.center(vacuum=4.0)

# Ground-state calculation
calc = GPAW(nbands=7, h=0.4,
            poissonsolver=PoissonSolver(eps=1e-16),
            convergence={'density': 1e-8},
            xc='GLLBSC',
            txt='gs.out')
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')

# Time-propagation calculation
td_calc = TDDFT('gs.gpw', txt='td.out')
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 3, 'dm.dat')

# Write a restart point
td_calc.write('td.gpw', mode='all')

# Keep propagating
td_calc.propagate(20, 3, 'dm.dat')

# Restart from the restart point
td_calc = TDDFT('td.gpw', txt='td2.out')
td_calc.propagate(20, 3, 'dm.dat')
world.barrier()

# Check dipole moment file
data_tj = np.loadtxt('dm.dat')
# Original run
ref_i = data_tj[4:6].ravel()
# Restarted steps
data_i = data_tj[7:].ravel()

tol = 1e-10
equal(data_i, ref_i, tol)

# Test the absolute values
data = np.loadtxt('dm.dat')[:6].ravel()
if 0:
    from gpaw.test import print_reference
    print_reference(data, 'ref', '%.12le')

ref = [0.0, 1.46915686e-15, -3.289312570937e-14, -2.273046460905e-14,
       -3.201827522804e-15, 0.82682747, 1.30253044e-15, 6.113786782415e-05,
       6.113753835177e-05, 6.113827464597e-05, 1.65365493, -1.69353262e-15,
       0.0001073089135539, 0.0001073052457949, 0.0001073068465827, 2.4804824,
       1.42934356e-15, 0.0001353894007493, 0.0001353887214486,
       0.0001353873226291, 3.30730987, -3.43926271e-16, 0.0001441529062519,
       0.000144155244532, 0.0001441536382364, 4.13413733, -8.41062896e-16,
       0.0001348222114923, 0.0001348264801341, 0.0001348229035149]

print(data.tolist())

tol = 1e-7
equal(data, ref, tol)
