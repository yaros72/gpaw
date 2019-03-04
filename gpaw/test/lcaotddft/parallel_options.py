import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.poisson import PoissonSolver
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.mpi import world

from gpaw.test import equal

# Atoms
atoms = molecule('NaCl')
atoms.center(vacuum=4.0)

# Ground-state calculation
calc = GPAW(nbands=7, h=0.4, setups=dict(Na='1'),
            basis='dzp', mode='lcao',
            poissonsolver=PoissonSolver(eps=1e-16),
            convergence={'density': 1e-8},
            txt='gs.out')
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')

# Reference time-propagation calculation
td_calc = LCAOTDDFT('gs.gpw', txt='td.out')
DipoleMomentWriter(td_calc, 'dm.dat')
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 3)

# Check dipole moment file
world.barrier()
ref = np.loadtxt('dm.dat').ravel()

# Test parallelization options
par_i = []

if world.size == 2:
    par_i.append({'band': 2})
    par_i.append({'sl_default': (2, 1, 2)})
    par_i.append({'sl_default': (1, 2, 4), 'band': 2})
elif world.size == 4:
    par_i.append({'band': 2})
    par_i.append({'sl_default': (2, 2, 2)})
    par_i.append({'sl_default': (2, 2, 4), 'band': 2})
else:
    par_i.append({'band': 2})
    par_i.append({'sl_auto': True})
    par_i.append({'sl_auto': True, 'band': 2})

for i, par in enumerate(par_i):
    td_calc = LCAOTDDFT('gs.gpw', parallel=par, txt='td%d.out' % i)
    DipoleMomentWriter(td_calc, 'dm%d.dat' % i)
    td_calc.absorption_kick(np.ones(3) * 1e-5)
    td_calc.propagate(20, 3)

    world.barrier()
    data = np.loadtxt('dm%d.dat' % i).ravel()

    tol = 1e-11
    equal(data, ref, tol)
