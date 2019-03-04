from ase.build import bulk
from gpaw import GPAW
from gpaw.response.df import DielectricFunction

# Part 1: Ground state calculation
atoms = bulk('Al', 'fcc', a=4.043)  # generate fcc crystal structure
# GPAW calculator initialization:
k = 13
calc = GPAW(mode='pw', kpts=(k, k, k))

atoms.set_calculator(calc)
atoms.get_potential_energy()  # ground state calculation is performed
calc.write('Al.gpw', 'all')  # use 'all' option to write wavefunctions

# Part 2: Spectrum calculation
df = DielectricFunction(calc='Al.gpw')  # ground state gpw file as input

# Momentum transfer, must be the difference between two kpoints:
q_c = [1.0 / k, 0, 0]
df.get_eels_spectrum(q_c=q_c)  # by default, an 'eels.csv' file is generated
