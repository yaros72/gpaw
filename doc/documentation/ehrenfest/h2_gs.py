from ase import Atoms
from gpaw import GPAW

name = 'h2_diss'

# Create H2 molecule in the center of a box aligned along the z axis.
d_bond = 0.754  # H2 equilibrium bond length
atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, d_bond)])
atoms.set_pbc(False)
atoms.center(vacuum=4.0)

# Set groundstate calculator and get and save wavefunctions
calc = GPAW(h=0.3, nbands=1, basis='dzp', txt=name + '_gs.txt')
atoms.set_calculator(calc)
atoms.get_potential_energy()
calc.write(name + '_gs.gpw', mode='all')
