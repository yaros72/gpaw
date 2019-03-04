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

fxc = 'LDA'
# Time-propagation calculation with fxc
td_calc = LCAOTDDFT('gs.gpw', fxc=fxc, txt='td_fxc.out')
DipoleMomentWriter(td_calc, 'dm_fxc.dat')
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 4)

# Time-propagation calculation with linearize_to_fxc()
td_calc = LCAOTDDFT('gs.gpw', txt='td_lin.out')
td_calc.linearize_to_xc(fxc)
DipoleMomentWriter(td_calc, 'dm_lin.dat')
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 4)

# Test the equivalence
world.barrier()
ref = np.loadtxt('dm_fxc.dat').ravel()
data = np.loadtxt('dm_lin.dat').ravel()

tol = 1e-9
equal(data, ref, tol)

# Test the absolute values
if 0:
    from gpaw.test import print_reference
    print_reference(data, 'ref', '%.12le')

ref = [0.0, 2.16436889e-15, -3.332079476367e-14, -2.770920572229e-14,
       -2.124228060013e-14, 0.0, -5.50792334e-16, -1.480925618326e-14,
       2.865764301706e-15, 1.117595510483e-15, 0.82682747, -1.2838216e-16,
       6.176332859889e-05, 6.176332861316e-05, 6.176332860951e-05,
       1.65365493, -3.74805176e-15, 9.885710868135e-05, 9.885710869956e-05,
       9.885710869476e-05, 2.4804824, -5.63543764e-17, 0.0001041392530612,
       0.0001041392530816, 0.0001041392530805, 3.30730987, -2.20860739e-15,
       8.817298882459e-05, 8.817298883414e-05, 8.817298882617e-05]

print('result')
print(data.tolist())

tol = 1e-12
equal(data, ref, tol)
