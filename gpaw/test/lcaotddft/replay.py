import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.poisson import PoissonSolver
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.lcaotddft.wfwriter import WaveFunctionWriter
from gpaw.tddft.spectrum import photoabsorption_spectrum
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
            txt='gs.out')
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')

# Time-propagation calculation
td_calc = LCAOTDDFT('gs.gpw', txt='td.out')
DipoleMomentWriter(td_calc, 'dm.dat')
WaveFunctionWriter(td_calc, 'wf.ulm')
WaveFunctionWriter(td_calc, 'wf_split.ulm', split=True)
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 3)
td_calc.write('td.gpw', mode='all')
td_calc.propagate(7, 3)

# Restart from the restart point
td_calc = LCAOTDDFT('td.gpw', txt='td2.out')
DipoleMomentWriter(td_calc, 'dm.dat')
WaveFunctionWriter(td_calc, 'wf.ulm')
WaveFunctionWriter(td_calc, 'wf_split.ulm')
td_calc.propagate(20, 3)
td_calc.propagate(20, 3)
td_calc.propagate(10, 3)
photoabsorption_spectrum('dm.dat', 'spec.dat')

world.barrier()
ref_i = np.loadtxt('spec.dat').ravel()

# Replay both wf*.ulm files
for tag in ['', '_split']:
    td_calc = LCAOTDDFT('gs.gpw', txt='rep%s.out' % tag)
    DipoleMomentWriter(td_calc, 'dm_rep%s.dat' % tag)
    td_calc.replay(name='wf%s.ulm' % tag, update='density')
    photoabsorption_spectrum('dm_rep%s.dat' % tag, 'spec_rep%s.dat' % tag)

    world.barrier()

    # Check the spectrum files
    # Do this instead of dipolemoment files in order to see that the kick
    # was also written correctly in replaying
    data_i = np.loadtxt('spec_rep%s.dat' % tag).ravel()

    tol = 1e-10
    equal(data_i, ref_i, tol)
