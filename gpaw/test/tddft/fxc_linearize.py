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

fxc = 'LDA'
# Time-propagation calculation with linearize_to_fxc()
td_calc = TDDFT('gs.gpw', txt='td.out')
td_calc.linearize_to_xc(fxc)
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 4, 'dm.dat')
world.barrier()

# Test the absolute values
data = np.loadtxt('dm.dat').ravel()
if 0:
    from gpaw.test import print_reference
    print_reference(data, 'ref', '%.12le')

ref = [0.0, 1.46915686e-15, -3.289312570937e-14, -2.273046460905e-14,
       -3.201827522804e-15, 0.82682747, 1.41057108e-15, 6.113786981692e-05,
       6.113754003915e-05, 6.113827654045e-05, 1.65365493, 1.69317502e-16,
       0.0001066406796495, 0.0001066481954317, 0.0001066442404437, 2.4804824,
       -1.16927902e-15, 0.0001341112381952, 0.0001341042452292,
       0.0001341080813738]

print(data.tolist())

tol = 1e-7
equal(data, ref, tol)
