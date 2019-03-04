"""Test automagical calculation of wfs"""

from gpaw import GPAW
from ase import Atoms

# H2
H = Atoms('HH', [(0, 0, 0), (0, 0, 1)])
H.center(vacuum=2.0)

calc = GPAW(nbands=2,
            convergence={'eigenstates': 1e-3})
H.set_calculator(calc)
H.get_potential_energy()
calc.write('tmp')

calc = GPAW('tmp')
calc.converge_wave_functions()

calc.set(nbands=5)
calc.converge_wave_functions()
