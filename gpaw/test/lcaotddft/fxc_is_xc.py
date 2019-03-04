import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.poisson import PoissonSolver
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.mpi import world

from gpaw.test import equal

# Atoms
atoms = molecule('Na2')
atoms.center(vacuum=4.0)

# Ground-state calculation
calc = GPAW(nbands=2, h=0.4, setups=dict(Na='1'),
            basis='dzp', mode='lcao',
            poissonsolver=PoissonSolver(eps=1e-16),
            convergence={'density': 1e-8},
            xc='LDA',
            txt='gs.out')
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')

# Time-propagation calculation without fxc
td_calc = LCAOTDDFT('gs.gpw', txt='td.out')
DipoleMomentWriter(td_calc, 'dm.dat')
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 4)
world.barrier()

# Check dipole moment file
ref = np.loadtxt('dm.dat').ravel()

# Time-propagation calculation with fxc=xc
td_calc = LCAOTDDFT('gs.gpw', fxc='LDA', txt='td_fxc.out')
DipoleMomentWriter(td_calc, 'dm_fxc.dat')
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 1)
td_calc.write('td_fxc.gpw', mode='all')

# Restart from the restart point
td_calc = LCAOTDDFT('td_fxc.gpw', txt='td_fxc2.out')
DipoleMomentWriter(td_calc, 'dm_fxc.dat')
td_calc.propagate(20, 3)

# Check dipole moment file
data = np.loadtxt('dm_fxc.dat')[[0, 1, 2, 4, 5, 6]].ravel()

tol = 1e-9
equal(data, ref, tol)
