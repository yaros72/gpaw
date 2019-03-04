from ase import Atoms
from gpaw import GPAW

H2 = Atoms('H2', positions=[(0, 0, 0), (1, 0, 0)])
H2.set_cell((3, 3.1, 3.2))
H2.center()
calc = GPAW(convergence={'forces': 0.01,
                         'density': 100,
                         'energy': 100,
                         'eigenstates': 100})
H2.set_calculator(calc)
H2.get_potential_energy()
n = calc.get_number_of_iterations()
assert 7 <= n <= 11, n
