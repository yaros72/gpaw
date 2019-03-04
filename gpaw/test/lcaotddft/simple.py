import numpy as np

from ase.build import molecule
from gpaw import GPAW
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.poisson import PoissonSolver
from gpaw.lcaotddft.dipolemomentwriter import DipoleMomentWriter
from gpaw.tddft.spectrum import photoabsorption_spectrum
from gpaw.mpi import world

from gpaw.test import equal

# Atoms
atoms = molecule('Na2')
atoms.center(vacuum=4.0)

# Ground-state calculation
calc = GPAW(nbands=2, h=0.4, setups=dict(Na='1'),
            basis='dzp', mode='lcao',
            poissonsolver=PoissonSolver('fd', eps=1e-16),
            convergence={'density': 1e-8},
            txt='gs.out')
atoms.set_calculator(calc)
energy = atoms.get_potential_energy()
calc.write('gs.gpw', mode='all')

# Time-propagation calculation
td_calc = LCAOTDDFT('gs.gpw', txt='td.out')
DipoleMomentWriter(td_calc, 'dm.dat')
td_calc.absorption_kick(np.ones(3) * 1e-5)
td_calc.propagate(20, 3)
photoabsorption_spectrum('dm.dat', 'spec.dat', delta_e=5)
world.barrier()

# Test dipole moment
data_i = np.loadtxt('dm.dat')[:, 2:].ravel()
if 0:
    from gpaw.test import print_reference
    print_reference(data_i, 'ref_i', '%.12le')

ref_i = [4.786589735249e-15,
         6.509942495725e-15,
         3.836848815869e-14,
         4.429061708370e-15,
         7.320865686028e-15,
         2.877243538173e-14,
         1.967175332445e-05,
         1.967175332505e-05,
         1.805003047148e-05,
         3.799528613595e-05,
         3.799528613766e-05,
         3.602504333467e-05,
         5.371491630029e-05,
         5.371491629857e-05,
         5.385043148270e-05]

tol = 1e-12
equal(data_i, ref_i, tol)

# Test spectrum
data_i = np.loadtxt('spec.dat').ravel()
if 0:
    from gpaw.test import print_reference
    print_reference(data_i, 'ref_i', '%.12le')

ref_i = [0.000000000000e+00,
         0.000000000000e+00,
         0.000000000000e+00,
         0.000000000000e+00,
         5.000000000000e+00,
         4.500226506100e-03,
         4.500226505900e-03,
         4.408376617800e-03,
         1.000000000000e+01,
         1.659425994100e-02,
         1.659425994100e-02,
         1.623811179300e-02,
         1.500000000000e+01,
         3.244686580500e-02,
         3.244686580400e-02,
         3.168680387500e-02,
         2.000000000000e+01,
         4.684883362500e-02,
         4.684883362300e-02,
         4.559686832900e-02,
         2.500000000000e+01,
         5.466780758300e-02,
         5.466780758100e-02,
         5.290167692700e-02,
         3.000000000000e+01,
         5.231585754700e-02,
         5.231585754600e-02,
         5.008658428800e-02]

tol = 1e-9
equal(data_i, ref_i, tol)
