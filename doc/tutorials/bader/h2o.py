from ase.build import molecule
from ase.io import write
from ase.units import Bohr
from gpaw import GPAW

atoms = molecule('H2O')
atoms.center(vacuum=3.5)
atoms.calc = GPAW(h=0.17, txt='h2o.txt')
atoms.get_potential_energy()
rho = atoms.calc.get_all_electron_density(gridrefinement=4)
write('density.cube', atoms, data=rho * Bohr**3)
