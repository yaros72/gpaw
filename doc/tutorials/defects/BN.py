import sys
import numpy as np
from ase import Atoms
from ase.build import niggli_reduce
from gpaw import GPAW, FermiDirac

# Script to get the total energies of a supercell
# of BN with and without a carbon substitution

c = 15.0

N = int(sys.argv[1])  # NxNx1 supercell
h = 0.15

a = 2.51026699
cell = [[a, 0., 0.],
        [-a / 2, np.sqrt(3) / 2 * a, 0.],
        [0., 0., c]]
scaled_positions = [[2. / 3, 1. / 3, 0.5],
                    [1. / 3, 2. / 3, 0.5]]

system = Atoms('BN',
               cell=cell,
               scaled_positions=scaled_positions,
               pbc=[1, 1, 1])
system = system.repeat((2, 1, 1))
niggli_reduce(system)

defect = system.repeat((N, N, 1))
defect[0].symbol = 'C'
defect[1].magmom = 1

q = +1  # Defect charge

calc = GPAW(mode='fd',
            kpts={'size': (4, 4, 1)},
            xc='PBE',
            charge=q,
            occupations=FermiDirac(0.01),
            txt='BN{0}{0}.CB_charged.txt'.format(N))


defect.set_calculator(calc)
defect.get_potential_energy()
calc.write('BN{0}{0}.C_B_charged.gpw'.format(N))

# Neutral case
parameters = calc.todict()
parameters['txt'] = 'BN{0}{0}.CB_neutral.txt'.format(N)
parameters['charge'] = 0
calc = GPAW(**parameters)

defect.set_calculator(calc)
defect.get_potential_energy()
calc.write('BN{0}{0}.C_B_neutral.gpw'.format(N))


# Now for the pristine case

pristine = system.repeat((N, N, 1))
parameters['txt'] = 'BN{0}{0}.pristine.txt'.format(N)
calc = GPAW(**parameters)

pristine.set_calculator(calc)
pristine.get_potential_energy()
calc.write('BN{0}{0}.pristine.gpw'.format(N))
