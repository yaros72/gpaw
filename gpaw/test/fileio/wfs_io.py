from ase import Atoms
from ase.units import Hartree
from gpaw import GPAW

h2 = Atoms('H2', [(0, 0, 0), (0, 0, 1)])
h2.center(vacuum=2.0)
calc = GPAW(nbands=2, convergence={'eigenstates': 1e-3})
h2.set_calculator(calc)
e0 = h2.get_potential_energy()
r0 = calc.wfs.eigensolver.error * Hartree**2 / 2
assert r0 < 1e-3
calc.write('h2', 'all')
    
# refine the restart file containing the wfs
calc = GPAW('h2', convergence={'eigenstates': 1e-5})
e1 = calc.get_atoms().get_potential_energy()
r1 = calc.wfs.eigensolver.error * Hartree**2 / 2
assert r1 < 1e-5
