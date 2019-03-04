import sys
from ase import Atoms
from gpaw import GPAW, FermiDirac

# Script to get the total energies of a supercell
# of GaAs with and without a Ga vacancy

a = 5.628  # Lattice parameter
N = int(sys.argv[1])  # NxNxN supercell
q = -3  # Defect charge

formula = 'Ga4As4'

lattice = [[a, 0.0, 0.0],  # work with cubic cell
           [0.0, a, 0.0],
           [0.0, 0.0, a]]

basis = [[0.0, 0.0, 0.0],
         [0.5, 0.5, 0.0],
         [0.0, 0.5, 0.5],
         [0.5, 0.0, 0.5],
         [0.25, 0.25, 0.25],
         [0.75, 0.75, 0.25],
         [0.25, 0.75, 0.75],
         [0.75, 0.25, 0.75]]

GaAs = Atoms(symbols=formula,
             scaled_positions=basis,
             cell=lattice,
             pbc=(1, 1, 1))

GaAsdef = GaAs.repeat((N, N, N))

GaAsdef.pop(0)  # Make the supercell and a Ga vacancy

calc = GPAW(mode='fd',
            kpts={'size': (2, 2, 2), 'gamma': False},
            xc='LDA',
            charge=q,
            occupations=FermiDirac(0.01),
            txt='GaAs{0}{0}{0}.Ga_vac.txt'.format(N))


GaAsdef.set_calculator(calc)
Edef = GaAsdef.get_potential_energy()

calc.write('GaAs{0}{0}{0}.Ga_vac_charged.gpw'.format(N))

# Now for the pristine case

GaAspris = GaAs.repeat((N, N, N))
parameters = calc.todict()
parameters['txt'] = 'GaAs{0}{0}{0}.pristine.txt'.format(N)
parameters['charge'] = 0
calc = GPAW(**parameters)

GaAspris.set_calculator(calc)
Epris = GaAspris.get_potential_energy()

calc.write('GaAs{0}{0}{0}.pristine.gpw'.format(N))
